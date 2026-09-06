"""
Voltage sensitivity matrix estimation, local linear approximation,
and validation against true OpenDSS AC power flow solutions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel

from grid_edge_research.config import ExperimentConfig, load_config
from grid_edge_research.data.feeder_mapping import DERPlacement, determine_der_placement
from grid_edge_research.powerflow.opendss_interface import OpenDSSInterface

logger = logging.getLogger(__name__)


class SensitivityModel(BaseModel):
    """
    Tractable local voltage-sensitivity approximation:
        V_approx = V_ref + S_P_batt * Delta_P_batt + S_P_pv * Delta_P_pv + S_Q_pv * Delta_Q_pv + S_P_load * Delta_P_load
    """
    node_names: List[str]
    pv_buses: List[str]
    battery_buses: List[str]
    
    # Reference operating point voltages (pu)
    v_ref: List[float]
    reference_load_kw: float
    reference_pv_pu: float
    
    # Sensitivity matrices (pu / kW or pu / kvar)
    # Shape: [len(node_names), len(der_buses)]
    s_p_pv: List[List[float]]       # (M, N_pv)
    s_q_pv: List[List[float]]       # (M, N_pv)
    s_p_batt: List[List[float]]     # (M, N_batt)
    s_p_load: List[float]           # (M,)
    
    perturbation_p_kw: float
    perturbation_q_kvar: float
    mae_validation_error: float = 0.0
    max_validation_error: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load(cls, path: str | Path) -> SensitivityModel:
        """Load sensitivity model from JSON file."""
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def predict_voltages(
        self,
        delta_p_batt_kw: np.ndarray | Dict[str, float],
        delta_p_pv_kw: np.ndarray | Dict[str, float],
        delta_q_pv_kvar: np.ndarray | Dict[str, float],
        delta_p_load_kw: float = 0.0,
    ) -> np.ndarray:
        """
        Predict node voltages using linear sensitivity model:
            V_pred = V_ref + S_P_batt * dP_batt + S_P_pv * dP_pv + S_Q_pv * dQ_pv + S_P_load * dP_load
        """
        v = np.array(self.v_ref, dtype=float).copy()

        # Vectorize inputs if provided as dicts
        if isinstance(delta_p_batt_kw, dict):
            dp_b = np.array([delta_p_batt_kw.get(b, 0.0) for b in self.battery_buses], dtype=float)
        else:
            dp_b = np.asarray(delta_p_batt_kw, dtype=float)

        if isinstance(delta_p_pv_kw, dict):
            dp_pv = np.array([delta_p_pv_kw.get(b, 0.0) for b in self.pv_buses], dtype=float)
        else:
            dp_pv = np.asarray(delta_p_pv_kw, dtype=float)

        if isinstance(delta_q_pv_kvar, dict):
            dq_pv = np.array([delta_q_pv_kvar.get(b, 0.0) for b in self.pv_buses], dtype=float)
        else:
            dq_pv = np.asarray(delta_q_pv_kvar, dtype=float)

        if len(dp_b) > 0 and len(self.s_p_batt) > 0:
            v += np.array(self.s_p_batt) @ dp_b

        if len(dp_pv) > 0 and len(self.s_p_pv) > 0:
            v += np.array(self.s_p_pv) @ dp_pv

        if len(dq_pv) > 0 and len(self.s_q_pv) > 0:
            v += np.array(self.s_q_pv) @ dq_pv

        if delta_p_load_kw != 0.0:
            v += np.array(self.s_p_load) * delta_p_load_kw

        return v


def estimate_feeder_sensitivities(
    config: Optional[ExperimentConfig] = None,
    placement: Optional[DERPlacement] = None,
    perturbation_p_kw: float = 25.0,
    perturbation_q_kvar: float = 25.0,
    save_path: Optional[Path | str] = None,
) -> SensitivityModel:
    """
    Estimate local voltage sensitivity matrices using finite perturbations in OpenDSS.
    """
    cfg = config or load_config()
    pl = placement or determine_der_placement(cfg)
    sim = OpenDSSInterface(cfg, pl)

    logger.info("Computing reference operating point for voltage sensitivities...")
    # Reference point: 1.0 pu nominal load, 0.5 pu PV, 0 battery
    sim.set_load_multiplier(1.0)
    sim.apply_der_controls(
        pv_curtailment_kw={},
        pv_q_kvar={},
        pv_available_pu=0.50,
        battery_p_kw={},
        dt_hours=1.0,
    )
    ref_state = sim.solve_power_flow()
    node_names = list(ref_state.bus_voltages_pu.keys())
    v_ref = np.array([ref_state.bus_voltages_pu[n] for n in node_names], dtype=float)

    # 1. Sensitivity w.r.t PV active power Delta P_pv
    s_p_pv_cols = []
    for bus in pl.pv_buses:
        # Perturb active injection by +delta P
        curt_dict = {bus: 0.0}
        sim.apply_der_controls(
            pv_curtailment_kw=curt_dict,
            pv_q_kvar={},
            pv_available_pu=0.50 + (perturbation_p_kw / max(1.0, pl.pv_ratings_kw.get(bus, 100.0))),
            battery_p_kw={},
            dt_hours=1.0,
        )
        p_state = sim.solve_power_flow()
        v_pert = np.array([p_state.bus_voltages_pu[n] for n in node_names], dtype=float)
        # S_P = (V_pert - V_ref) / dP
        # Find realized injection delta at this bus
        dp = p_state.pv_generation_kw.get(bus, perturbation_p_kw) - ref_state.pv_generation_kw.get(bus, 0.0)
        dp = dp if abs(dp) > 1e-3 else perturbation_p_kw
        sens_col = (v_pert - v_ref) / dp
        s_p_pv_cols.append(sens_col)

    s_p_pv = np.column_stack(s_p_pv_cols) if s_p_pv_cols else np.zeros((len(node_names), 0))

    # 2. Sensitivity w.r.t PV reactive power Delta Q_pv
    s_q_pv_cols = []
    for bus in pl.pv_buses:
        q_dict = {bus: perturbation_q_kvar}
        sim.apply_der_controls(
            pv_curtailment_kw={},
            pv_q_kvar=q_dict,
            pv_available_pu=0.50,
            battery_p_kw={},
            dt_hours=1.0,
        )
        q_state = sim.solve_power_flow()
        v_pert = np.array([q_state.bus_voltages_pu[n] for n in node_names], dtype=float)
        dq = q_state.pv_reactive_kvar.get(bus, perturbation_q_kvar) - ref_state.pv_reactive_kvar.get(bus, 0.0)
        dq = dq if abs(dq) > 1e-3 else perturbation_q_kvar
        sens_col = (v_pert - v_ref) / dq
        s_q_pv_cols.append(sens_col)

    s_q_pv = np.column_stack(s_q_pv_cols) if s_q_pv_cols else np.zeros((len(node_names), 0))

    # 3. Sensitivity w.r.t Battery active power Delta P_batt
    s_p_batt_cols = []
    for bus in pl.battery_buses:
        # Reset battery SOC high enough to discharge
        sim.battery_soc[bus] = 0.50
        b_dict = {bus: perturbation_p_kw}
        sim.apply_der_controls(
            pv_curtailment_kw={},
            pv_q_kvar={},
            pv_available_pu=0.50,
            battery_p_kw=b_dict,
            dt_hours=1.0,
        )
        b_state = sim.solve_power_flow()
        v_pert = np.array([b_state.bus_voltages_pu[n] for n in node_names], dtype=float)
        dp = b_state.battery_p_kw.get(bus, perturbation_p_kw) - ref_state.battery_p_kw.get(bus, 0.0)
        dp = dp if abs(dp) > 1e-3 else perturbation_p_kw
        sens_col = (v_pert - v_ref) / dp
        s_p_batt_cols.append(sens_col)

    s_p_batt = np.column_stack(s_p_batt_cols) if s_p_batt_cols else np.zeros((len(node_names), 0))

    # 4. Sensitivity w.r.t aggregate Feeder Load
    sim.set_load_multiplier(1.05)
    sim.apply_der_controls(
        pv_curtailment_kw={},
        pv_q_kvar={},
        pv_available_pu=0.50,
        battery_p_kw={},
        dt_hours=1.0,
    )
    l_state = sim.solve_power_flow()
    v_pert = np.array([l_state.bus_voltages_pu[n] for n in node_names], dtype=float)
    dp_load = (1.05 - 1.00) * 3490.0  # ~174.5 kW
    s_p_load = (v_pert - v_ref) / dp_load

    # Reset simulator
    sim.reset_to_initial_state()

    # 5. Validation on separate random perturbations
    logger.info("Validating local sensitivity approximation against random OpenDSS test dispatches...")
    rng = np.random.RandomState(999)
    errors = []
    for _ in range(cfg.sensitivity.validation_sample_count):
        test_load_pu = float(rng.uniform(0.6, 1.2))
        test_pv_pu = float(rng.uniform(0.0, 1.0))
        test_batt = {b: float(rng.uniform(-100.0, 100.0)) for b in pl.battery_buses}
        test_q = {b: float(rng.uniform(-50.0, 50.0)) for b in pl.pv_buses}
        test_curt = {b: float(rng.uniform(0.0, 20.0)) for b in pl.pv_buses}

        # True OpenDSS AC solve
        sim.set_load_multiplier(test_load_pu)
        sim.apply_der_controls(
            pv_curtailment_kw=test_curt,
            pv_q_kvar=test_q,
            pv_available_pu=test_pv_pu,
            battery_p_kw=test_batt,
            dt_hours=1.0,
        )
        ac_state = sim.solve_power_flow()
        v_true = np.array([ac_state.bus_voltages_pu[n] for n in node_names], dtype=float)

        # Approximate Sensitivity prediction
        dp_batt_arr = np.array([test_batt.get(b, 0.0) for b in pl.battery_buses])
        dp_pv_arr = np.array([(test_pv_pu - 0.50) * pl.pv_ratings_kw.get(b, 0.0) - test_curt.get(b, 0.0) for b in pl.pv_buses])
        dq_pv_arr = np.array([test_q.get(b, 0.0) for b in pl.pv_buses])
        dp_load_val = (test_load_pu - 1.00) * 3490.0

        v_pred = (
            v_ref
            + (s_p_batt @ dp_batt_arr if len(dp_batt_arr) > 0 else 0)
            + (s_p_pv @ dp_pv_arr if len(dp_pv_arr) > 0 else 0)
            + (s_q_pv @ dq_pv_arr if len(dq_pv_arr) > 0 else 0)
            + s_p_load * dp_load_val
        )
        errors.append(np.abs(v_true - v_pred))

    all_errors = np.concatenate(errors)
    mae_err = float(np.mean(all_errors))
    max_err = float(np.max(all_errors))
    logger.info(f"Sensitivity Model Validation Error: MAE={mae_err:.6f} pu, Max={max_err:.6f} pu")

    model = SensitivityModel(
        node_names=node_names,
        pv_buses=pl.pv_buses,
        battery_buses=pl.battery_buses,
        v_ref=[float(x) for x in v_ref],
        reference_load_kw=3490.0,
        reference_pv_pu=0.50,
        s_p_pv=[[float(x) for x in row] for row in s_p_pv],
        s_q_pv=[[float(x) for x in row] for row in s_q_pv],
        s_p_batt=[[float(x) for x in row] for row in s_p_batt],
        s_p_load=[float(x) for x in s_p_load],
        perturbation_p_kw=float(perturbation_p_kw),
        perturbation_q_kvar=float(perturbation_q_kvar),
        mae_validation_error=mae_err,
        max_validation_error=max_err,
    )

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
        logger.info(f"Saved sensitivity model to {out_p}")

    return model
