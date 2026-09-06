"""
Engineering Rule-Based Baseline Controller:
Implements transparent IEEE 1547-style Volt-VAR droop control,
heuristic solar-excess battery charging, peak-load battery discharge,
and emergency overvoltage PV curtailment.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import numpy as np

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.controllers.base import BaseController, DERControlAction
from grid_edge_research.data.feeder_mapping import DERPlacement
from grid_edge_research.powerflow.network_state import NetworkState
from grid_edge_research.uncertainty.scenario_generator import ScenarioSet


class RuleBasedController(BaseController):
    """
    Transparent Engineering Baseline:
    1. Smart-Inverter Volt-VAR: IEEE 1547 piecewise droop curve.
    2. Battery Dispatch: Charge during peak solar generation; discharge during evening peak load.
    3. PV Curtailment: Emergency curtailment if local bus voltage exceeds 1.05 pu.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        placement: DERPlacement,
    ):
        super().__init__(config, placement, controller_name="rule_based")

    def compute_action(
        self,
        current_state: NetworkState,
        forecast_load_kw: np.ndarray,
        forecast_pv_pu: np.ndarray,
        scenarios: Optional[ScenarioSet] = None,
        timestamp: Optional[str] = None,
    ) -> DERControlAction:
        t_start = time.perf_counter()
        pv_q: Dict[str, float] = {}
        pv_curt: Dict[str, float] = {}
        batt_p: Dict[str, float] = {}

        curr_pv_pu = float(forecast_pv_pu[0]) if len(forecast_pv_pu) > 0 else 0.0
        curr_load_kw = float(forecast_load_kw[0]) if len(forecast_load_kw) > 0 else 2000.0

        # 1. Volt-VAR Inverter Control for each PV bus
        for bus in self.placement.pv_buses:
            p_rated = self.placement.pv_ratings_kw.get(bus, 0.0)
            s_rated = p_rated * self.config.pv.inverter.s_oversize_factor
            p_avail = p_rated * curr_pv_pu
            
            # Available reactive power headroom: sqrt(S_max^2 - P^2)
            q_max_avail = np.sqrt(max(0.0, s_rated**2 - p_avail**2))

            # Find local bus voltage (matching bus string prefix)
            bus_v_candidates = [v for k, v in current_state.bus_voltages_pu.items() if k.startswith(f"{bus}.")]
            v_local = np.mean(bus_v_candidates) if bus_v_candidates else current_state.v_mean_pu

            # Volt-VAR piecewise droop rule
            # Overvoltage (V > 1.03): absorb reactive power (Q < 0)
            # Undervoltage (V < 0.97): inject reactive power (Q > 0)
            # Deadband (0.97 <= V <= 1.03): Q = 0
            if v_local > 1.03:
                # Droop slope between 1.03 and 1.05
                droop_ratio = min(1.0, (v_local - 1.03) / 0.02)
                q_cmd = -droop_ratio * q_max_avail
            elif v_local < 0.97:
                droop_ratio = min(1.0, (0.97 - v_local) / 0.02)
                q_cmd = droop_ratio * q_max_avail
            else:
                q_cmd = 0.0

            pv_q[bus] = float(q_cmd)

            # Emergency PV curtailment if local voltage exceeds 1.05
            if v_local > 1.05:
                curt_ratio = min(1.0, (v_local - 1.05) / 0.03)
                pv_curt[bus] = float(curt_ratio * p_avail)
            else:
                pv_curt[bus] = 0.0

        # 2. Battery Dispatch Rule
        # Charge when solar is abundant (pv_pu > 0.40)
        # Discharge when load is high and solar is low (pv_pu < 0.15 and load > 2200 kW)
        batt_cfg = self.config.battery
        for bus in self.placement.battery_buses:
            p_cap = self.placement.battery_ratings_kw.get(bus, 100.0)
            curr_soc = current_state.battery_soc.get(bus, batt_cfg.soc_initial)

            if curr_pv_pu >= 0.35 and curr_soc < (batt_cfg.soc_max - 0.02):
                # Charge (P_batt < 0)
                charge_power = min(p_cap, p_cap * (curr_pv_pu / 0.8))
                batt_p[bus] = -float(charge_power)
            elif curr_pv_pu < 0.20 and curr_load_kw > 2000.0 and curr_soc > (batt_cfg.soc_min + 0.05):
                # Discharge (P_batt > 0)
                discharge_power = min(p_cap, p_cap * ((curr_load_kw - 2000.0) / 1500.0))
                batt_p[bus] = float(discharge_power)
            else:
                batt_p[bus] = 0.0

        t_elapsed = time.perf_counter() - t_start
        return DERControlAction(
            pv_curtailment_kw=pv_curt,
            pv_q_kvar=pv_q,
            battery_p_kw=batt_p,
            battery_q_kvar={},
            solver_status="optimal",
            solve_time_seconds=t_elapsed,
            fallback_used=False,
            controller_name="rule_based",
        )
