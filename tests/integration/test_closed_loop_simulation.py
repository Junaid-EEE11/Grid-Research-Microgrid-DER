"""
Integration tests for end-to-end multi-hour closed-loop simulation and metrics generation.
"""

import json
from pathlib import Path
import pytest

from scripts.run_experiment import run_experiment


def test_closed_loop_multi_hour_smoke_experiment(tmp_path):
    """Run a 6-hour closed loop smoke simulation and verify output integrity."""
    sens_path = Path("results/models/sensitivity_model.json")
    if not sens_path.exists():
        pytest.skip("Sensitivity model required for closed loop test.")

    out_dir = tmp_path / "test_runs"
    run_id = "test_smoke_closed_loop"

    run_path = run_experiment(
        controller_name="rule_based",
        pv_penetration=0.50,
        uncertainty_level="nominal",
        n_steps=6,
        split="test",
        run_id=run_id,
        output_dir=str(out_dir),
    )

    summary_file = run_path / "summary_metrics.json"
    assert summary_file.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_hours_simulated"] == 6
    assert summary["opendss_convergence_failures_count"] == 0
    assert summary["max_voltage_pu"] >= summary["min_voltage_pu"]
    assert summary["curtailment_percentage"] >= 0.0

    step_records_p = run_path / "step_records.csv"
    assert step_records_p.exists()
