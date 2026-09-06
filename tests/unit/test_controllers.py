"""
Unit tests for controller interfaces, output schemas, and deterministic fallback.
"""

from pathlib import Path
import numpy as np
import pytest

from grid_edge_research.config import load_config
from grid_edge_research.controllers.base import DERControlAction
from grid_edge_research.controllers.deterministic_mpc import DeterministicMPCController
from grid_edge_research.controllers.rule_based import RuleBasedController
from grid_edge_research.controllers.stochastic_mpc import StochasticMPCController
from grid_edge_research.data.feeder_mapping import determine_der_placement
from grid_edge_research.optimization.sensitivity_model import SensitivityModel
from grid_edge_research.powerflow.network_state import NetworkState

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def test_setup():
    cfg = load_config()
    placement = determine_der_placement(cfg)
    sens_path = REPO_ROOT / "results" / "models" / "sensitivity_model.json"
    if not sens_path.exists():
        pytest.skip(f"Sensitivity model artifact not found at {sens_path}")
    sens = SensitivityModel.load(str(sens_path))

    state = NetworkState(
        timestamp="2014-06-01 12:00:00",
        converged=True,
        iterations=3,
        bus_voltages_pu={"114.1": 1.02, "48.1": 1.01},
        v_min_pu=1.01,
        v_max_pu=1.02,
        v_mean_pu=1.015,
        total_violation_exposure=0.0,
        nodes_violating_count=0,
        total_node_count=2,
        substation_p_kw=1000.0,
        substation_q_kvar=200.0,
        feeder_losses_kw=50.0,
        feeder_losses_kvar=10.0,
        battery_soc={b: 0.50 for b in placement.battery_buses},
        battery_p_kw={b: 0.0 for b in placement.battery_buses},
        pv_generation_kw={b: 50.0 for b in placement.pv_buses},
        pv_reactive_kvar={b: 0.0 for b in placement.pv_buses},
        pv_curtailment_kw={b: 0.0 for b in placement.pv_buses},
    )

    load_forecast = np.full(24, 1200.0)
    pv_forecast = np.full(24, 0.80)

    return cfg, placement, sens, state, load_forecast, pv_forecast


def test_rule_based_controller_schema(test_setup):
    cfg, placement, sens, state, load_forecast, pv_forecast = test_setup
    ctrl = RuleBasedController(cfg, placement)
    action = ctrl.compute_action(state, load_forecast, pv_forecast)

    assert isinstance(action, DERControlAction)
    assert len(action.pv_curtailment_kw) == len(placement.pv_buses)
    assert len(action.pv_q_kvar) == len(placement.pv_buses)
    assert len(action.battery_p_kw) == len(placement.battery_buses)
    assert action.solver_status in ["optimal", "heuristic_ok"]


def test_deterministic_mpc_controller_schema(test_setup):
    cfg, placement, sens, state, load_forecast, pv_forecast = test_setup
    ctrl = DeterministicMPCController(cfg, placement, sens)
    action = ctrl.compute_action(state, load_forecast, pv_forecast)

    assert isinstance(action, DERControlAction)
    assert len(action.pv_curtailment_kw) == len(placement.pv_buses)
    assert len(action.battery_p_kw) == len(placement.battery_buses)
    assert action.solver_status in ["optimal", "optimal_inaccurate"]


def test_stochastic_mpc_controller_schema(test_setup):
    cfg, placement, sens, state, load_forecast, pv_forecast = test_setup
    ctrl = StochasticMPCController(cfg, placement, sens)
    action = ctrl.compute_action(state, load_forecast, pv_forecast)

    assert isinstance(action, DERControlAction)
    assert len(action.pv_curtailment_kw) == len(placement.pv_buses)
    assert len(action.battery_p_kw) == len(placement.battery_buses)
    assert action.solver_status in ["optimal", "optimal_inaccurate"]
