"""
Unit tests for multi-step scenario generator and calibration modifiers.
"""

import numpy as np
import pytest

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.uncertainty.scenario_generator import ScenarioGenerator


def test_scenario_generation_bounds_and_reproducibility():
    """Verify scenarios obey non-negativity and reproducibility with fixed random seed."""
    cfg = ExperimentConfig()
    gen = ScenarioGenerator(cfg)

    # Populate synthetic residual pool
    np.random.seed(42)
    fake_residuals = np.random.normal(loc=0, scale=10.0, size=(100, 24))
    gen.load_residuals = fake_residuals
    gen.pv_residuals = fake_residuals * 0.01

    load_median = np.full(24, 500.0)
    pv_median = np.full(24, 0.60)

    scens_1 = gen.generate_scenarios(load_median, pv_median, scenario_count=10, step_seed=123)
    scens_2 = gen.generate_scenarios(load_median, pv_median, scenario_count=10, step_seed=123)

    # Check reproducibility
    assert np.allclose(scens_1.load_scenarios_kw, scens_2.load_scenarios_kw)
    assert np.allclose(scens_1.pv_scenarios_pu, scens_2.pv_scenarios_pu)

    # Check physical constraints: load >= 0, 0 <= pv <= 1
    assert np.all(np.array(scens_1.load_scenarios_kw) >= 0.0)
    assert np.all(np.array(scens_1.pv_scenarios_pu) >= 0.0)
    assert np.all(np.array(scens_1.pv_scenarios_pu) <= 1.0)


def test_scenario_calibration_modes():
    """Verify under-dispersed and over-dispersed scenario calibrations."""
    cfg = ExperimentConfig()
    gen = ScenarioGenerator(cfg)
    gen.load_residuals = np.ones((50, 24)) * 10.0
    gen.pv_residuals = np.ones((50, 24)) * 0.1

    load_median = np.full(24, 100.0)
    pv_median = np.full(24, 0.5)

    scens_nom = gen.generate_scenarios(load_median, pv_median, scenario_count=5, calibration_mode="well_calibrated")
    scens_under = gen.generate_scenarios(load_median, pv_median, scenario_count=5, calibration_mode="under_dispersed")
    scens_over = gen.generate_scenarios(load_median, pv_median, scenario_count=5, calibration_mode="over_dispersed")

    # In our deterministic pool, check direct delta
    assert np.mean(np.array(scens_under.load_scenarios_kw) - 100.0) < np.mean(np.array(scens_nom.load_scenarios_kw) - 100.0)
    assert np.mean(np.array(scens_over.load_scenarios_kw) - 100.0) > np.mean(np.array(scens_nom.load_scenarios_kw) - 100.0)
