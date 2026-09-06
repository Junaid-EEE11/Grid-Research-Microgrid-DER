"""
Deterministic Model Predictive Controller (MPC):
Optimizes DER active/reactive schedules over a 24-hour horizon using point/median forecasts
and the critical-bus OpenDSS voltage sensitivity model via fast CLARABEL / HIGHS solves.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
import numpy as np

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.controllers.base import BaseController, DERControlAction
from grid_edge_research.controllers.rule_based import RuleBasedController
from grid_edge_research.data.feeder_mapping import DERPlacement
from grid_edge_research.optimization.sensitivity_model import SensitivityModel
from grid_edge_research.powerflow.network_state import NetworkState
from grid_edge_research.uncertainty.scenario_generator import ScenarioSet

logger = logging.getLogger(__name__)


class DeterministicMPCController(BaseController):
    """
    Fast Deterministic MPC:
    Formulates a convex program over horizon H=24 using the top voltage-sensitive nodes.
    Applies only the first-step control decision in receding horizon fashion.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        placement: DERPlacement,
        sensitivity_model: SensitivityModel,
        n_critical_nodes: int = 25,
    ):
        super().__init__(config, placement, controller_name="deterministic_mpc")
        self.sens = sensitivity_model
        self.rule_fallback = RuleBasedController(config, placement)

        # Select top most sensitive nodes for fast and robust LP/QP solve
        sp_full = np.array(self.sens.s_p_pv)       # (M, N_pv)
        sq_full = np.array(self.sens.s_q_pv)       # (M, N_pv)
        sb_full = np.array(self.sens.s_p_batt)     # (M, N_batt)
        s_load_full = np.array(self.sens.s_p_load) # (M,)
        v_ref_full = np.array(self.sens.v_ref)     # (M,)

        sens_score = (
            np.max(np.abs(sp_full), axis=1)
            + np.max(np.abs(sq_full), axis=1)
            + (np.max(np.abs(sb_full), axis=1) if sb_full.shape[1] > 0 else 0)
            + np.abs(s_load_full)
        )
        n_crit = min(n_critical_nodes, len(sens_score))
        crit_idx = np.argsort(-sens_score)[:n_crit]

        self.S_pv_p = sp_full[crit_idx, :]
        self.S_pv_q = sq_full[crit_idx, :]
        self.S_batt_p = sb_full[crit_idx, :]
        self.S_load_p = s_load_full[crit_idx]
        self.v_ref = v_ref_full[crit_idx]

        self.n_nodes = n_crit
        self.n_pv = len(self.placement.pv_buses)
        self.n_batt = len(self.placement.battery_buses)

        self.pv_ratings_vec = np.array([self.placement.pv_ratings_kw.get(b, 0.0) for b in self.placement.pv_buses])
        self.s_pv_ratings_vec = self.pv_ratings_vec * self.config.pv.inverter.s_oversize_factor
        self.batt_p_max = np.array([self.placement.battery_ratings_kw.get(b, 100.0) for b in self.placement.battery_buses])
        self.batt_e_cap = np.array([self.placement.battery_ratings_kwh.get(b, 200.0) for b in self.placement.battery_buses])

    def compute_action(
        self,
        current_state: NetworkState,
        forecast_load_kw: np.ndarray,
        forecast_pv_pu: np.ndarray,
        scenarios: Optional[ScenarioSet] = None,
        timestamp: Optional[str] = None,
    ) -> DERControlAction:
        t_start = time.perf_counter()
        H = min(self.config.forecasting.forecast_horizon_hours, len(forecast_load_kw), len(forecast_pv_pu))
        dt = self.config.forecasting.control_interval_hours
        weights = self.config.objective_weights
        v_lim = self.config.voltage_limits
        batt_cfg = self.config.battery

        try:
            # 1. Decision Variables (Matrix Shape H x Dim)
            p_curt = cp.Variable((H, self.n_pv), nonneg=True)
            q_pv = cp.Variable((H, self.n_pv))
            p_dis = cp.Variable((H, self.n_batt), nonneg=True)
            p_ch = cp.Variable((H, self.n_batt), nonneg=True)
            soc = cp.Variable((H + 1, self.n_batt))
            u_viol = cp.Variable((H, self.n_nodes), nonneg=True)
            l_viol = cp.Variable((H, self.n_nodes), nonneg=True)

            constraints = []

            # 2. PV Availability & Inverter Box Constraints
            p_avail_matrix = np.outer(forecast_pv_pu[:H], self.pv_ratings_vec)  # (H, N_pv)
            constraints.append(p_curt <= p_avail_matrix)
            p_used = p_avail_matrix - p_curt

            if self.config.optimization.enable_reactive_power:
                q_box = np.outer(np.ones(H), self.s_pv_ratings_vec * 0.44)
                constraints.append(q_pv <= q_box)
                constraints.append(q_pv >= -q_box)
            else:
                constraints.append(q_pv == 0.0)

            # 3. Battery Constraints & SOC Dynamics (Vectorized)
            init_soc_vec = np.array([current_state.battery_soc.get(b, batt_cfg.soc_initial) for b in self.placement.battery_buses])
            constraints.append(soc[0, :] == init_soc_vec)

            if self.config.optimization.enable_battery:
                p_max_mat = np.outer(np.ones(H), self.batt_p_max)
                constraints.append(p_dis <= p_max_mat)
                constraints.append(p_ch <= p_max_mat)

                e_dis_mat = np.outer(np.ones(H), self.batt_e_cap * batt_cfg.eta_discharge)
                e_ch_mat = np.outer(np.ones(H), self.batt_e_cap / batt_cfg.eta_charge)

                constraints.append(
                    soc[1:, :] == soc[:-1, :] - cp.multiply(p_dis, dt / e_dis_mat) + cp.multiply(p_ch, dt / e_ch_mat)
                )
                constraints.append(soc >= batt_cfg.soc_min)
                constraints.append(soc <= batt_cfg.soc_max)
            else:
                constraints.append(p_dis == 0.0)
                constraints.append(p_ch == 0.0)
                constraints.append(soc == np.outer(np.ones(H + 1), init_soc_vec))

            # 4. Vectorized Voltage Sensitivity Model on Critical Nodes
            p_batt_net = p_dis - p_ch  # (H, N_batt)
            dp_pv = p_used - np.outer(np.ones(H), 0.50 * self.pv_ratings_vec)  # (H, N_pv)
            dp_load = (forecast_load_kw[:H] - self.sens.reference_load_kw)[:, None]  # (H, 1)

            v_ref_mat = np.outer(np.ones(H), self.v_ref)
            V = (
                v_ref_mat
                + (dp_pv @ self.S_pv_p.T)
                + (q_pv @ self.S_pv_q.T)
                + (p_batt_net @ self.S_batt_p.T)
                + (dp_load @ self.S_load_p[None, :])
            )

            constraints.append(u_viol >= V - v_lim.v_max_pu)
            constraints.append(l_viol >= v_lim.v_min_pu - V)

            # 5. Total Objective
            cost = (
                weights.w_voltage_violation * cp.sum(u_viol + l_viol)
                + weights.w_pv_curtailment * cp.sum(p_curt)
                + weights.w_battery_wear * cp.sum(p_dis + p_ch)
                + weights.w_reactive_power * cp.sum(cp.abs(q_pv))
            )

            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(solver=cp.SCIPY, verbose=False)

            if prob.status not in ["optimal", "optimal_inaccurate"] or p_curt.value is None:
                logger.warning(f"Deterministic MPC solve status '{prob.status}'. Engaging rule-based fallback.")
                fallback_action = self.rule_fallback.compute_action(
                    current_state, forecast_load_kw, forecast_pv_pu, scenarios, timestamp
                )
                fallback_action.solver_status = str(prob.status)
                fallback_action.fallback_used = True
                fallback_action.controller_name = "deterministic_mpc_fallback"
                return fallback_action

            # Extract first-step control decisions (t=0)
            pv_curt_0 = {bus: float(max(0.0, p_curt.value[0, i])) for i, bus in enumerate(self.placement.pv_buses)}
            pv_q_0 = {bus: float(q_pv.value[0, i]) for i, bus in enumerate(self.placement.pv_buses)}
            batt_p_0 = {
                bus: float(p_dis.value[0, b] - p_ch.value[0, b])
                for b, bus in enumerate(self.placement.battery_buses)
            }

            t_elapsed = time.perf_counter() - t_start
            return DERControlAction(
                pv_curtailment_kw=pv_curt_0,
                pv_q_kvar=pv_q_0,
                battery_p_kw=batt_p_0,
                battery_q_kvar={},
                solver_status=str(prob.status),
                solve_time_seconds=t_elapsed,
                fallback_used=False,
                controller_name="deterministic_mpc",
            )

        except Exception as e:
            logger.error(f"Deterministic MPC exception: {e}. Falling back to rule-based.")
            fallback_action = self.rule_fallback.compute_action(
                current_state, forecast_load_kw, forecast_pv_pu, scenarios, timestamp
            )
            fallback_action.solver_status = "exception"
            fallback_action.fallback_used = True
            fallback_action.controller_name = "deterministic_mpc_fallback"
            return fallback_action
