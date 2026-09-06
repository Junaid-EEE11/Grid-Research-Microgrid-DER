"""
Closed-loop receding horizon physical simulation engine.
Coordinates forecasting, uncertainty generation, MPC control optimization,
OpenDSS AC power flow execution, and metric extraction across multi-day horizons.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.controllers.base import BaseController, DERControlAction
from grid_edge_research.data.feeder_mapping import DERPlacement, determine_der_placement
from grid_edge_research.metrics.grid_metrics import ExperimentMetrics, compute_simulation_metrics
from grid_edge_research.powerflow.network_state import NetworkState
from grid_edge_research.powerflow.opendss_interface import OpenDSSInterface
from grid_edge_research.uncertainty.scenario_generator import ScenarioGenerator, ScenarioSet

logger = logging.getLogger(__name__)


class ClosedLoopSimulator:
    """
    Simulates physical AC power flow closed-loop receding horizon control on the IEEE 123 feeder.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        controller: BaseController,
        placement: Optional[DERPlacement] = None,
        scenario_generator: Optional[ScenarioGenerator] = None,
        forecast_models: Optional[Dict] = None,
    ):
        self.config = config
        self.controller = controller
        self.placement = placement or determine_der_placement(config)
        self.scenario_gen = scenario_generator or ScenarioGenerator(config)
        self.forecast_models = forecast_models or {}
        self.dss_sim = OpenDSSInterface(self.config, self.placement)

    def run_simulation(
        self,
        feature_df: pd.DataFrame,
        start_idx: int = 0,
        n_steps: Optional[int] = None,
        experiment_id: str = "run_001",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, ExperimentMetrics]:
        """
        Execute full closed-loop simulation over feature_df rows from start_idx to start_idx + n_steps.
        """
        total_len = len(feature_df)
        H = self.config.forecasting.forecast_horizon_hours
        steps_to_run = min(n_steps or (total_len - start_idx - H), total_len - start_idx - H)

        if steps_to_run <= 0:
            raise ValueError(f"Insufficient time steps to run simulation with horizon {H}.")

        logger.info(f"Starting closed-loop simulation for {steps_to_run} steps with controller '{self.controller.controller_name}'...")
        self.dss_sim.reset_to_initial_state()

        # Initial power flow solve to establish initial state
        init_load_pu = float(feature_df["load_pu"].iloc[start_idx])
        self.dss_sim.set_load_multiplier(init_load_pu)
        curr_state = self.dss_sim.solve_power_flow(str(feature_df.index[start_idx]))

        step_records: List[Dict] = []
        iterator = range(start_idx, start_idx + steps_to_run)
        if verbose:
            iterator = tqdm(iterator, desc=f"Simulating ({self.controller.controller_name})")

        for step_i in iterator:
            t_stamp = str(feature_df.index[step_i])
            window_df = feature_df.iloc[step_i : step_i + H]

            # 1. Multi-step forecasts (H=24)
            # Load forecast
            if "load_kw_quantile_gbr" in self.forecast_models:
                l_preds = self.forecast_models["load_kw_quantile_gbr"].predict(window_df)[0.50]
            elif "load_kw_point_gbr" in self.forecast_models:
                l_preds = self.forecast_models["load_kw_point_gbr"].predict(window_df)
            else:
                # Perfect foresight or persistence fallback
                l_preds = window_df["load_kw"].values

            # PV forecast
            if "pv_pu_quantile_gbr" in self.forecast_models:
                p_preds = self.forecast_models["pv_pu_quantile_gbr"].predict(window_df)[0.50]
            elif "pv_pu_point_gbr" in self.forecast_models:
                p_preds = self.forecast_models["pv_pu_point_gbr"].predict(window_df)
            else:
                p_preds = window_df["pv_pu"].values

            # 2. Multi-step Uncertainty Scenarios
            scenarios = None
            if "stochastic" in self.controller.controller_name:
                scenarios = self.scenario_gen.generate_scenarios(
                    load_median=l_preds,
                    pv_median=p_preds,
                    solar_zenith=window_df["solar_zenith"].values if "solar_zenith" in window_df.columns else None,
                    scenario_count=self.config.uncertainty.scenario_count,
                    inflation_factor=self.config.uncertainty.uncertainty_inflation_factor,
                    calibration_mode=self.config.uncertainty.calibration_mode,
                    step_seed=self.config.random_seed + step_i,
                )

            # 3. Compute Control Action
            action: DERControlAction = self.controller.compute_action(
                current_state=curr_state,
                forecast_load_kw=l_preds,
                forecast_pv_pu=p_preds,
                scenarios=scenarios,
                timestamp=t_stamp,
            )

            # 4. Apply Realized Load and PV Injections in OpenDSS
            realized_load_pu = float(feature_df["load_pu"].iloc[step_i])
            realized_pv_pu = float(feature_df["pv_pu"].iloc[step_i])

            self.dss_sim.set_load_multiplier(realized_load_pu)
            self.dss_sim.apply_der_controls(
                pv_curtailment_kw=action.pv_curtailment_kw,
                pv_q_kvar=action.pv_q_kvar,
                pv_available_pu=realized_pv_pu,
                battery_p_kw=action.battery_p_kw,
                battery_q_kvar=action.battery_q_kvar,
                dt_hours=self.config.forecasting.control_interval_hours,
            )

            # 5. Physical OpenDSS AC Power Flow Solve
            curr_state = self.dss_sim.solve_power_flow(t_stamp)

            # 6. Record step performance
            pv_gen_sum = sum(curr_state.pv_generation_kw.values())
            pv_curt_sum = sum(action.pv_curtailment_kw.values())
            batt_abs_sum = sum(abs(p) for p in curr_state.battery_p_kw.values())
            mean_soc = float(np.mean(list(curr_state.battery_soc.values()))) if curr_state.battery_soc else 0.50

            record = {
                "timestamp": t_stamp,
                "converged": curr_state.converged,
                "iterations": curr_state.iterations,
                "v_min_pu": curr_state.v_min_pu,
                "v_max_pu": curr_state.v_max_pu,
                "v_mean_pu": curr_state.v_mean_pu,
                "total_violation_exposure": curr_state.total_violation_exposure,
                "nodes_violating_count": curr_state.nodes_violating_count,
                "total_node_count": curr_state.total_node_count,
                "substation_p_kw": curr_state.substation_p_kw,
                "substation_q_kvar": curr_state.substation_q_kvar,
                "feeder_losses_kw": curr_state.feeder_losses_kw,
                "pv_gen_total_kw": pv_gen_sum,
                "pv_curt_total_kw": pv_curt_sum,
                "battery_abs_power_kw": batt_abs_sum,
                "battery_mean_soc": mean_soc,
                "solver_status": action.solver_status,
                "solve_time_seconds": action.solve_time_seconds,
                "fallback_used": action.fallback_used,
            }
            step_records.append(record)

        records_df = pd.DataFrame(step_records)
        records_df["timestamp"] = pd.to_datetime(records_df["timestamp"])
        records_df.set_index("timestamp", inplace=True)

        # Aggregate total battery capacity across all batteries
        tot_batt_kwh = sum(self.placement.battery_ratings_kwh.values())

        summary_metrics = compute_simulation_metrics(
            step_records_df=records_df,
            experiment_id=experiment_id,
            controller=self.controller.controller_name,
            pv_penetration_pct=self.config.pv.penetration_level * 100.0,
            uncertainty_level=self.config.uncertainty.stress_test_mode,
            nominal_battery_capacity_kwh=tot_batt_kwh,
        )

        return records_df, summary_metrics
