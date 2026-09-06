"""
High-level OpenDSS interface for unbalanced AC power flow simulation,
DER dispatch injection, load scaling, and network state extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import opendssdirect as dss

from grid_edge_research.config import ExperimentConfig, load_config
from grid_edge_research.data.feeder_mapping import DERPlacement, determine_der_placement
from grid_edge_research.powerflow.feeder_setup import get_bus_phase_spec
from grid_edge_research.powerflow.network_state import NetworkState

logger = logging.getLogger(__name__)


class OpenDSSInterface:
    """
    Manages the physical OpenDSS simulation of the IEEE 123-bus feeder.
    Provides rigorous physical AC power flow execution, state extraction,
    and DER control enforcement.
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        placement: Optional[DERPlacement] = None,
    ):
        self.config = config or load_config()
        self.placement = placement or determine_der_placement(self.config)
        feeder_path = Path(self.config.paths.feeder_dir)
        if (feeder_path / "IEEE123Master.dss").exists():
            self.feeder_dir = feeder_path.resolve()
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            if (repo_root / "feeder" / "ieee123" / "IEEE123Master.dss").exists():
                self.feeder_dir = (repo_root / "feeder" / "ieee123").resolve()
            elif (repo_root / feeder_path).exists():
                self.feeder_dir = (repo_root / feeder_path).resolve()
            else:
                self.feeder_dir = feeder_path.resolve()
        self.master_file = self.feeder_dir / "IEEE123Master.dss"

        if not self.master_file.exists():
            raise FileNotFoundError(
                f"Master DSS file not found at {self.master_file}. Run scripts/fetch_feeder.py first."
            )

        # Internal state tracking
        self.battery_soc: Dict[str, float] = {
            b: self.config.battery.soc_initial for b in self.placement.battery_buses
        }
        self.initialized = False
        self._init_circuit()

    def _init_circuit(self) -> None:
        """Compile base feeder and instantiate controllable DER objects."""
        dss.Text.Command("Clear")
        dss.Text.Command(f'Compile "{self.master_file}"')

        # Fix regulator control mode to isolate DER response during simulation
        dss.Text.Command("Set ControlMode=OFF")

        # 1. Add PV generators
        for bus, rating_kw in self.placement.pv_ratings_kw.items():
            if rating_kw > 0:
                bus_spec, phases, kv = get_bus_phase_spec(bus)
                s_rated = rating_kw * self.config.pv.inverter.s_oversize_factor
                cmd = (
                    f"New Generator.PV_{bus} Bus1={bus_spec} Phases={phases} kV={kv} kVA={s_rated:.2f} "
                    f"kW={rating_kw:.2f} kvar=0.0 Model=1 Vminpu=0.80 Vmaxpu=1.20"
                )
                dss.Text.Command(cmd)

        # 2. Add Battery storage elements (modeled as bidirectional Generators)
        for bus in self.placement.battery_buses:
            bus_spec, phases, kv = get_bus_phase_spec(bus)
            p_rating = self.placement.battery_ratings_kw[bus]
            s_rating = p_rating * 1.10
            cmd = (
                f"New Generator.BATT_{bus} Bus1={bus_spec} Phases={phases} kV={kv} kVA={s_rating:.2f} "
                f"kW=0.0 kvar=0.0 Model=1 Vminpu=0.80 Vmaxpu=1.20"
            )
            dss.Text.Command(cmd)

        dss.Text.Command("Solve")
        if not dss.Solution.Converged():
            logger.warning("Initial OpenDSS solve did not converge.")
        self.initialized = True

    def reset_to_initial_state(self) -> None:
        """Reset network and battery state of charge to initial values."""
        self.battery_soc = {
            b: self.config.battery.soc_initial for b in self.placement.battery_buses
        }
        self._init_circuit()

    def set_load_multiplier(self, load_pu: float) -> None:
        """Scale all background loads uniformly according to time-series load multiplier."""
        dss.Text.Command(f"Set LoadMult={load_pu:.6f}")

    def apply_der_controls(
        self,
        pv_curtailment_kw: Dict[str, float],
        pv_q_kvar: Dict[str, float],
        pv_available_pu: float,
        battery_p_kw: Dict[str, float],
        battery_q_kvar: Optional[Dict[str, float]] = None,
        dt_hours: float = 1.0,
    ) -> None:
        """
        Apply control setpoints to PV inverters and batteries, update battery SOC.
        
        Sign conventions:
            P_batt > 0: discharging (injecting into grid)
            P_batt < 0: charging (consuming from grid)
            Q > 0: capacitive / injecting reactive power
            Q < 0: inductive / absorbing reactive power
        """
        # 1. Update PV generators
        for bus, nameplate_kw in self.placement.pv_ratings_kw.items():
            if nameplate_kw <= 0:
                continue
            gen_name = f"Generator.PV_{bus}"
            p_avail = nameplate_kw * pv_available_pu
            p_curtail = pv_curtailment_kw.get(bus, 0.0)
            p_curtail = np.clip(p_curtail, 0.0, p_avail)
            p_injected = p_avail - p_curtail

            q_injected = pv_q_kvar.get(bus, 0.0) if self.config.optimization.enable_reactive_power else 0.0
            
            # Enforce inverter apparent power limit S^2 = P^2 + Q^2 <= S_max^2
            s_max = nameplate_kw * self.config.pv.inverter.s_oversize_factor
            s_sq = p_injected**2 + q_injected**2
            if s_sq > s_max**2 and s_sq > 0:
                scale = s_max / np.sqrt(s_sq)
                p_injected *= scale
                q_injected *= scale

            dss.Circuit.SetActiveElement(gen_name)
            dss.Properties.Value("kW", f"{p_injected:.4f}")
            dss.Properties.Value("kvar", f"{q_injected:.4f}")

        # 2. Update Batteries and SOC dynamics
        batt_cfg = self.config.battery
        for bus in self.placement.battery_buses:
            gen_name = f"Generator.BATT_{bus}"
            p_cmd = battery_p_kw.get(bus, 0.0) if self.config.optimization.enable_battery else 0.0
            p_max = self.placement.battery_ratings_kw[bus]
            e_cap = self.placement.battery_ratings_kwh[bus]

            # Current SOC
            curr_soc = self.battery_soc[bus]

            # Physical power limits based on SOC boundaries
            max_discharge = min(p_max, max(0.0, (curr_soc - batt_cfg.soc_min) * e_cap * batt_cfg.eta_discharge / dt_hours))
            max_charge = min(p_max, max(0.0, (batt_cfg.soc_max - curr_soc) * e_cap / (batt_cfg.eta_charge * dt_hours)))

            p_realized = np.clip(p_cmd, -max_charge, max_discharge)

            # Update SOC
            if p_realized >= 0:  # Discharge
                soc_next = curr_soc - (p_realized * dt_hours) / (e_cap * batt_cfg.eta_discharge)
            else:  # Charge
                soc_next = curr_soc + (-p_realized * dt_hours * batt_cfg.eta_charge) / e_cap

            self.battery_soc[bus] = float(np.clip(soc_next, batt_cfg.soc_min, batt_cfg.soc_max))

            q_realized = 0.0
            if battery_q_kvar and self.config.optimization.enable_reactive_power:
                q_realized = battery_q_kvar.get(bus, 0.0)

            dss.Circuit.SetActiveElement(gen_name)
            dss.Properties.Value("kW", f"{p_realized:.4f}")
            dss.Properties.Value("kvar", f"{q_realized:.4f}")

    def solve_power_flow(self, timestamp: Optional[str] = None) -> NetworkState:
        """Run AC power flow in OpenDSS and extract comprehensive physical network state."""
        dss.Text.Command("Solve")
        converged = bool(dss.Solution.Converged())
        iterations = int(dss.Solution.Iterations())

        # Extract node voltages
        node_names = list(dss.Circuit.AllNodeNames())
        v_mag_pu_array = np.array(dss.Circuit.AllBusMagPu())

        voltages_pu: Dict[str, float] = {}
        for name, vpu in zip(node_names, v_mag_pu_array):
            voltages_pu[name] = float(vpu)

        v_min = float(np.min(v_mag_pu_array)) if len(v_mag_pu_array) > 0 else 1.0
        v_max = float(np.max(v_mag_pu_array)) if len(v_mag_pu_array) > 0 else 1.0
        v_mean = float(np.mean(v_mag_pu_array)) if len(v_mag_pu_array) > 0 else 1.0

        v_lim_min = self.config.voltage_limits.v_min_pu
        v_lim_max = self.config.voltage_limits.v_max_pu

        # Total violation exposure = sum over all nodes of [max(0, V - V_max) + max(0, V_min - V)]
        under_viol = np.maximum(0.0, v_lim_min - v_mag_pu_array)
        over_viol = np.maximum(0.0, v_mag_pu_array - v_lim_max)
        total_viol = float(np.sum(under_viol + over_viol))
        viol_nodes = int(np.sum((under_viol > 1e-4) | (over_viol > 1e-4)))

        # Substation Power
        total_p_losses, total_q_losses = dss.Circuit.Losses()  # Watts, Vars
        total_losses_kw = total_p_losses / 1000.0
        total_losses_kvar = total_q_losses / 1000.0

        total_power = dss.Circuit.TotalPower()  # kW, kvar
        substation_p_kw = -float(total_power[0])
        substation_q_kvar = -float(total_power[1])

        # Extract realized DER injections
        pv_gen: Dict[str, float] = {}
        pv_curt: Dict[str, float] = {}
        pv_q: Dict[str, float] = {}
        for bus in self.placement.pv_buses:
            gen_name = f"Generator.PV_{bus}"
            dss.Circuit.SetActiveElement(gen_name)
            powers = dss.CktElement.Powers()
            if powers and len(powers) >= 2:
                p_gen = -sum(powers[0::2])
                q_gen = -sum(powers[1::2])
                pv_gen[bus] = float(p_gen)
                pv_q[bus] = float(q_gen)

        batt_p: Dict[str, float] = {}
        for bus in self.placement.battery_buses:
            gen_name = f"Generator.BATT_{bus}"
            dss.Circuit.SetActiveElement(gen_name)
            powers = dss.CktElement.Powers()
            if powers and len(powers) >= 2:
                p_out = -sum(powers[0::2])
                batt_p[bus] = float(p_out)

        return NetworkState(
            timestamp=timestamp,
            converged=converged,
            iterations=iterations,
            bus_voltages_pu=voltages_pu,
            v_min_pu=v_min,
            v_max_pu=v_max,
            v_mean_pu=v_mean,
            total_violation_exposure=total_viol,
            nodes_violating_count=viol_nodes,
            total_node_count=len(node_names),
            substation_p_kw=substation_p_kw,
            substation_q_kvar=substation_q_kvar,
            feeder_losses_kw=total_losses_kw,
            feeder_losses_kvar=total_losses_kvar,
            pv_generation_kw=pv_gen,
            pv_curtailment_kw=pv_curt,
            pv_reactive_kvar=pv_q,
            battery_p_kw=batt_p,
            battery_soc=dict(self.battery_soc),
        )
