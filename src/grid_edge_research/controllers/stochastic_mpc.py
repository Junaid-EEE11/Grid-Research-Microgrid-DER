"""
Risk-Aware Stochastic Model Predictive Controller (Stochastic MPC):
Coordinates battery active power, smart-inverter reactive power, and PV curtailment
across multi-step uncertainty scenarios using Conditional Value-at-Risk (CVaR)
for voltage-violation tail risk mitigation with critical-node CLARABEL / SCIPY solves.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
import numpy as np

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.controllers.base import BaseController, DERControlAction
from grid_edge_research.controllers.deterministic_mpc import DeterministicMPCController
from grid_edge_research.controllers.rule_based import RuleBasedController
from grid_edge_research.data.feeder_mapping import DERPlacement
from grid_edge_research.optimization.sensitivity_model import SensitivityModel
from grid_edge_research.powerflow.network_state import NetworkState
from grid_edge_research.uncertainty.scenario_generator import ScenarioGenerator, ScenarioSet

logger = logging.getLogger(__name__)


class StochasticMPCController(BaseController):
    """
    DCP-Compliant Fast Risk-Aware Stochastic MPC:
    Optimizes expected operational performance + CVaR_alpha voltage tail risk
    over N multi-step load and solar scenarios via fast critical-node QP/LP.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        placement: DERPlacement,
        sensitivity_model: SensitivityModel,
        scenario_generator: Optional[ScenarioGenerator] = None,
        n_critical_nodes: int = 25,
    ):
        super().__init__(config, placement, controller_name="stochastic_mpc")
        self.sens = sensitivity_model
        self.scenario_gen = scenario_generator or ScenarioGenerator(config)
        self.det_fallback = DeterministicMPCController(config, placement, sensitivity_model, n_critical_nodes)
        self.rule_fallback = RuleBasedController(config, placement)

        # Select top most sensitive nodes for fast and accurate QP solve
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
        alpha = self.config.uncertainty.cvar_alpha
        w_cvar = weights.w_cvar_risk

        # 1. Multi-step scenarios
        if scenarios is None:
            scenarios = self.scenario_gen.generate_scenarios(
                load_median=forecast_load_kw[:H],
                pv_median=forecast_pv_pu[:H],
                scenario_count=self.config.uncertainty.scenario_count,
                inflation_factor=self.config.uncertainty.uncertainty_inflation_factor,
                calibration_mode=self.config.uncertainty.calibration_mode,
            )

        N_scen = min(10, scenarios.n_scenarios)  # Fast tractable scenario count
        load_scens = np.array(scenarios.load_scenarios_kw)[:N_scen, :H]  # (N_scen, H)
        pv_scens = np.array(scenarios.pv_scenarios_pu)[:N_scen, :H]      # (N_scen, H)

        try:
            # First-stage decision variables (H x Dim)
            p_curt = cp.Variable((H, self.n_pv), nonneg=True)
            q_pv = cp.Variable((H, self.n_pv))
            p_dis = cp.Variable((H, self.n_batt), nonneg=True)
            p_ch = cp.Variable((H, self.n_batt), nonneg=True)
            soc = cp.Variable((H + 1, self.n_batt))

            # CVaR Auxiliary Variables
            gamma = cp.Variable(1)
            z = cp.Variable(N_scen, nonneg=True)

            constraints = []

            # 2. PV Availability & Inverter Box Constraints
            p_nom_avail = np.outer(forecast_pv_pu[:H], self.pv_ratings_vec)
            min_scen_avail = np.min(pv_scens, axis=0)  # (H,)
            p_min_avail_mat = np.outer(min_scen_avail, self.pv_ratings_vec)
            constraints.append(p_curt <= p_min_avail_mat)

            if self.config.optimization.enable_reactive_power:
                q_box = np.outer(np.ones(H), self.s_pv_ratings_vec * 0.44)
                constraints.append(q_pv <= q_box)
                constraints.append(q_pv >= -q_box)
            else:
                constraints.append(q_pv == 0.0)

            # 3. Battery Constraints & Vectorized SOC
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

            # 4. Vectorized Scenario Evaluations & CVaR Constraints (Purely Affine)
            p_batt_net = p_dis - p_ch  # (H, N_batt)
            v_ref_mat = np.outer(np.ones(H), self.v_ref)
            scenario_violation_losses = []

            for s in range(N_scen):
                u_s = cp.Variable((H, self.n_nodes), nonneg=True)
                l_s = cp.Variable((H, self.n_nodes), nonneg=True)

                p_avail_s = np.outer(pv_scens[s, :], self.pv_ratings_vec)
                p_used_s = p_avail_s - p_curt  # Affine expression
                dp_pv_s = p_used_s - np.outer(np.ones(H), 0.50 * self.pv_ratings_vec)
                dp_load_s = (load_scens[s, :] - self.sens.reference_load_kw)[:, None]

                V_s = (
                    v_ref_mat
                    + (dp_pv_s @ self.S_pv_p.T)
                    + (q_pv @ self.S_pv_q.T)
                    + (p_batt_net @ self.S_batt_p.T)
                    + (dp_load_s @ self.S_load_p[None, :])
                )

                constraints.append(u_s >= V_s - v_lim.v_max_pu)
                constraints.append(l_s >= v_lim.v_min_pu - V_s)

                s_loss = cp.sum(u_s + l_s)
                scenario_violation_losses.append(s_loss)
                constraints.append(z[s] >= s_loss - gamma[0])

            # CVaR Risk Term: CVaR_alpha = gamma + (1 / ((1 - alpha) * N)) * sum(z_s)
            cvar_risk = gamma[0] + (1.0 / ((1.0 - alpha) * N_scen)) * cp.sum(z)

            # Expected Cost
            e_viol_cost = (weights.w_voltage_violation / N_scen) * cp.sum(scenario_violation_losses)
            curt_cost = weights.w_pv_curtailment * cp.sum(p_curt)
            batt_cost = weights.w_battery_wear * cp.sum(p_dis + p_ch)
            q_cost = weights.w_reactive_power * cp.sum(cp.abs(q_pv))

            total_obj = cp.Minimize(e_viol_cost + curt_cost + batt_cost + q_cost + (w_cvar * cvar_risk))

            prob = cp.Problem(total_obj, constraints)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                prob.solve(solver=cp.SCIPY, verbose=False)

            if prob.status not in ["optimal", "optimal_inaccurate"] or p_curt.value is None:
                logger.warning(f"Stochastic MPC status '{prob.status}'. Attempting deterministic MPC fallback.")
                det_action = self.det_fallback.compute_action(
                    current_state, forecast_load_kw, forecast_pv_pu, scenarios, timestamp
                )
                det_action.fallback_used = True
                det_action.controller_name = "stochastic_fallback_to_deterministic"
                return det_action

            # Extract first-step control decision (t=0)
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
                controller_name="stochastic_mpc",
            )

        except Exception as e:
            logger.error(f"Stochastic MPC exception: {e}. Falling back to deterministic MPC.")
            det_action = self.det_fallback.compute_action(
                current_state, forecast_load_kw, forecast_pv_pu, scenarios, timestamp
            )
            det_action.fallback_used = True
            det_action.controller_name = "stochastic_fallback_to_deterministic"
            return det_action
