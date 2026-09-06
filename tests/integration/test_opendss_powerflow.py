"""
Integration tests for OpenDSS feeder circuit setup and 3-phase AC power flow solve.
"""

from pathlib import Path
import pytest

from grid_edge_research.config import load_config
from grid_edge_research.data.feeder_mapping import determine_der_placement
from grid_edge_research.powerflow.opendss_interface import OpenDSSInterface


def test_opendss_circuit_compiles_and_solves():
    """Verify IEEE 123 OpenDSS circuit compiles, converges, and returns voltage vector."""
    cfg = load_config()
    placement = determine_der_placement(cfg)

    dss = OpenDSSInterface(cfg, placement)

    # Step 1: Solve nominal condition
    dss.set_load_multiplier(1.0)
    dss.apply_der_controls(
        pv_curtailment_kw={b: 0.0 for b in placement.pv_buses},
        pv_q_kvar={b: 0.0 for b in placement.pv_buses},
        pv_available_pu=0.80,
        battery_p_kw={b: 0.0 for b in placement.battery_buses},
        dt_hours=1.0,
    )
    state = dss.solve_power_flow(timestamp="2014-06-01 12:00:00")

    assert state.converged is True
    assert state.substation_p_kw > 0.0
    assert len(state.bus_voltages_pu) > 200
    # Voltages should be reasonable per-unit numbers around 1.0
    for node, v_pu in state.bus_voltages_pu.items():
        assert 0.80 <= v_pu <= 1.20, f"Node {node} voltage {v_pu} out of bounds"
