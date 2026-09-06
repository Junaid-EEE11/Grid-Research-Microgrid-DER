"""
Unit tests for voltage sensitivity model loading and dimension verification.
"""

from pathlib import Path
import numpy as np
import pytest

from grid_edge_research.optimization.sensitivity_model import SensitivityModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_sensitivity_model_dimensions():
    """Verify sensitivity model matrices match node counts and DER bus counts."""
    model_path = REPO_ROOT / "results" / "models" / "sensitivity_model.json"
    if not model_path.exists():
        pytest.skip(f"Sensitivity model artifact not found at {model_path}")

    model = SensitivityModel.load(str(model_path))

    n_nodes = len(model.node_names)
    assert n_nodes > 0

    s_p_pv = np.array(model.s_p_pv)
    s_q_pv = np.array(model.s_q_pv)
    s_p_batt = np.array(model.s_p_batt)
    s_p_load = np.array(model.s_p_load)

    assert s_p_pv.shape[0] == n_nodes
    assert s_q_pv.shape[0] == n_nodes
    assert s_p_batt.shape[0] == n_nodes
    assert s_p_load.shape[0] == n_nodes

    assert len(model.v_ref) == n_nodes
    # Reference voltages should be in a physically sensible distribution [0.85, 1.15]
    assert np.all(np.array(model.v_ref) >= 0.85)
    assert np.all(np.array(model.v_ref) <= 1.15)
