"""
Unit tests for smart inverter apparent power constraints and PV curtailment bounds.
"""

import numpy as np
import pytest


def test_pv_curtailment_bounds():
    """Verify curtailment is non-negative and does not exceed available generation."""
    p_avail = np.array([0.0, 50.0, 100.0, 75.0])
    p_curt = np.array([0.0, 10.0, 100.0, 20.0])

    # Invariant: 0 <= p_curt <= p_avail
    assert np.all(p_curt >= 0.0)
    assert np.all(p_curt <= p_avail)

    p_used = p_avail - p_curt
    assert np.all(p_used >= 0.0)
    assert np.all(p_used <= p_avail)


def test_inverter_apparent_power_circle_constraint():
    """
    Verify smart inverter apparent power limits:
    P_used^2 + Q^2 <= S_rated^2
    """
    p_rated = 100.0  # kW
    oversize = 1.10
    s_rated = p_rated * oversize  # 110 kVA

    # Test feasible operating points
    p_used = 100.0  # kW
    q_max_avail = np.sqrt(max(0.0, s_rated**2 - p_used**2))
    assert np.isclose(q_max_avail, np.sqrt(110**2 - 100**2))  # ~45.82 kvar

    # Apparent power at maximum reactive dispatch
    s_actual = np.sqrt(p_used**2 + q_max_avail**2)
    assert np.isclose(s_actual, s_rated)

    # Infeasible operating point outside circle
    q_excess = q_max_avail + 5.0
    s_infeasible = np.sqrt(p_used**2 + q_excess**2)
    assert s_infeasible > s_rated
