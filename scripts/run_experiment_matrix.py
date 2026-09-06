"""
Comprehensive experiment matrix and ablation study executor.
Runs the complete matrix of controllers, PV penetrations, uncertainty inflations,
ablation variants, and forecast calibration configurations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from scripts.run_experiment import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_matrix(
    n_steps: int = 168,  # Default 1-week (168 hours) benchmark across conditions
    split: str = "test",
    output_dir: str = "results/runs",
    summary_table_path: str = "results/tables/experiment_matrix_summary.csv",
) -> pd.DataFrame:
    """
    Execute all combinations in the experiment matrix and record tidy tabular results.
    """
    out_base = Path(output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    table_p = Path(summary_table_path).resolve()
    table_p.parent.mkdir(parents=True, exist_ok=True)

    controllers = ["rule_based", "deterministic_mpc", "stochastic_mpc"]
    pv_penetrations = [0.0, 0.25, 0.50, 0.75, 1.00]
    uncertainty_levels = ["nominal", "moderate", "strong"]

    # 1. Primary Grid: Controllers x PV Penetrations (at nominal uncertainty)
    experiments_to_run = []
    for ctrl in controllers:
        for pen in pv_penetrations:
            run_id = f"{ctrl}_pv{int(pen*100)}_nominal"
            experiments_to_run.append({
                "controller": ctrl,
                "pv_penetration": pen,
                "uncertainty": "nominal",
                "run_id": run_id,
                "config_overrides": {},
            })

    # 2. Uncertainty Stress Tests (at 50% PV penetration)
    for ctrl in controllers:
        for unc in ["moderate", "strong"]:
            run_id = f"{ctrl}_pv50_{unc}"
            experiments_to_run.append({
                "controller": ctrl,
                "pv_penetration": 0.50,
                "uncertainty": unc,
                "run_id": run_id,
                "config_overrides": {},
            })

    # 3. Ablation Studies (at 50% PV penetration, nominal uncertainty)
    ablations = [
        ("ablation_no_q_control", "stochastic_mpc", {"optimization.enable_reactive_power": False}),
        ("ablation_no_battery", "stochastic_mpc", {"optimization.enable_battery": False}),
        ("ablation_zero_cvar_weight", "stochastic_mpc", {"optimization.weights.w_cvar_risk": 0.0}),
        ("ablation_under_dispersed", "stochastic_mpc", {"uncertainty.calibration_mode": "under_dispersed"}),
        ("ablation_over_dispersed", "stochastic_mpc", {"uncertainty.calibration_mode": "over_dispersed"}),
    ]

    for run_id, ctrl, overrides in ablations:
        experiments_to_run.append({
            "controller": ctrl,
            "pv_penetration": 0.50,
            "uncertainty": "nominal",
            "run_id": run_id,
            "config_overrides": overrides,
        })

    logger.info(f"Total experiments in study matrix: {len(experiments_to_run)}")
    records = []

    for item in tqdm(experiments_to_run, desc="Running Experiment Matrix"):
        run_id = item["run_id"]
        run_folder = out_base / run_id
        summary_json = run_folder / "summary_metrics.json"

        if summary_json.exists():
            logger.info(f"Skipping already completed run '{run_id}'")
            with open(summary_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append(data)
            continue

        # Load config and apply any specific ablation overrides
        cfg = load_config()
        cfg.controller = item["controller"]
        cfg.pv.penetration_level = item["pv_penetration"]
        cfg.uncertainty.stress_test_mode = item["uncertainty"]

        for k, v in item["config_overrides"].items():
            parts = k.split(".")
            target = cfg
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], v)

        # Save temporary YAML for this run
        tmp_cfg_path = run_folder / "config.yaml"
        tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(tmp_cfg_path)

        run_experiment(
            config_path=str(tmp_cfg_path),
            controller_name=item["controller"],
            pv_penetration=item["pv_penetration"],
            uncertainty_level=item["uncertainty"],
            n_steps=n_steps,
            split=split,
            run_id=run_id,
            output_dir=output_dir,
        )

        with open(summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        records.append(data)

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(table_p, index=False)
    logger.info(f"Experiment matrix complete. Summary table saved to {table_p}")
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete grid experiment matrix.")
    parser.add_argument("--steps", type=int, default=168, help="Number of hours to simulate per condition (default: 168)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="results/runs")

    args = parser.parse_args()
    run_full_matrix(n_steps=args.steps, split=args.split, output_dir=args.output_dir)
