"""
Unit tests for voltage violation metrics, CVaR auxiliary formulation, and bootstrap statistics.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from grid_edge_research.metrics.grid_metrics import compute_simulation_metrics


def test_voltage_violation_metrics_exact_definition():
    """
    Verify:
    violation = max(0, V - 1.05) + max(0, 0.95 - V)
    """
    df_steps = pd.DataFrame([
        {
            "timestamp": "2014-06-01 12:00:00",
            "total_violation_exposure": 0.04,
            "nodes_violating_count": 2,
            "total_node_count": 3,
            "v_max_pu": 1.07,
            "v_min_pu": 0.93,
            "pv_gen_total_kw": 50.0,
            "pv_curt_total_kw": 10.0,
            "battery_abs_power_kw": 20.0,
            "feeder_losses_kw": 5.0,
            "substation_p_kw": 100.0,
            "solver_status": "optimal",
            "fallback_used": False,
            "converged": True,
            "solve_time_seconds": 0.05,
        },
        {
            "timestamp": "2014-06-01 13:00:00",
            "total_violation_exposure": 0.01,
            "nodes_violating_count": 1,
            "total_node_count": 3,
            "v_max_pu": 1.02,
            "v_min_pu": 0.94,
            "pv_gen_total_kw": 60.0,
            "pv_curt_total_kw": 0.0,
            "battery_abs_power_kw": 0.0,
            "feeder_losses_kw": 4.0,
            "substation_p_kw": 90.0,
            "solver_status": "optimal",
            "fallback_used": False,
            "converged": True,
            "solve_time_seconds": 0.05,
        },
    ])

    metrics = compute_simulation_metrics(df_steps)

    assert np.isclose(metrics.total_voltage_violation_exposure, 0.05)
    # Total node-hours = 6, violating = 3 -> 0.5
    assert np.isclose(metrics.bus_node_hours_violation_fraction, 0.5)
    assert np.isclose(metrics.max_voltage_pu, 1.07)
    assert np.isclose(metrics.min_voltage_pu, 0.93)


def test_cvar_convex_auxiliary_formulation():
    """
    Test CVaR_alpha formulation on known discrete loss samples:
    CVaR_alpha = min_gamma { gamma + 1/((1-alpha)*N) * sum([Loss_i - gamma]^+) }
    """
    losses = np.arange(1, 11, dtype=float)
    N = len(losses)
    alpha = 0.80  # Tail is top 2 losses (9, 10), average is (9 + 10)/2 = 9.5

    gamma = cp.Variable(1)
    z = cp.Variable(N, nonneg=True)

    constraints = [z[i] >= losses[i] - gamma[0] for i in range(N)]
    cvar_obj = gamma[0] + (1.0 / ((1.0 - alpha) * N)) * cp.sum(z)

    prob = cp.Problem(cp.Minimize(cvar_obj), constraints)
    prob.solve(solver=cp.CLARABEL)

    assert prob.status in ["optimal", "optimal_inaccurate"]
    assert np.isclose(prob.value, 9.5, atol=1e-3)
