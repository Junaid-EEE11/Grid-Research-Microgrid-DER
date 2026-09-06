"""
Single-run experiment CLI runner.
Loads configuration, initializes controller and physical OpenDSS simulation,
runs the closed-loop receding-horizon MPC simulation, and persists full artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from grid_edge_research.controllers.deterministic_mpc import DeterministicMPCController
from grid_edge_research.controllers.rule_based import RuleBasedController
from grid_edge_research.controllers.stochastic_mpc import StochasticMPCController
from grid_edge_research.data.feeder_mapping import determine_der_placement
from grid_edge_research.data.loader import get_chronological_splits, load_processed_data
from grid_edge_research.forecasting.features import build_forecasting_features
from grid_edge_research.optimization.sensitivity_model import SensitivityModel
from grid_edge_research.simulation.closed_loop import ClosedLoopSimulator
from grid_edge_research.uncertainty.scenario_generator import ScenarioGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(
    config_path: str | None = None,
    controller_name: str | None = None,
    pv_penetration: float | None = None,
    uncertainty_level: str | None = None,
    n_steps: int | None = None,
    split: str = "test",
    run_id: str | None = None,
    output_dir: str = "results/runs",
) -> Path:
    """Execute a single closed-loop experiment run and save artifacts."""
    cfg = load_config(config_path)

    if controller_name:
        cfg.controller = controller_name
    if pv_penetration is not None:
        cfg.pv.penetration_level = pv_penetration
    if uncertainty_level:
        cfg.uncertainty.stress_test_mode = uncertainty_level
        if uncertainty_level == "moderate":
            cfg.uncertainty.uncertainty_inflation_factor = 1.50
        elif uncertainty_level == "strong":
            cfg.uncertainty.uncertainty_inflation_factor = 2.00
        else:
            cfg.uncertainty.uncertainty_inflation_factor = 1.00

    exp_id = run_id or f"{cfg.controller}_pv{int(cfg.pv.penetration_level*100)}_{cfg.uncertainty.stress_test_mode}"
    run_dir = Path(output_dir).resolve() / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and features
    raw_df = load_processed_data(cfg)
    feat_df = build_forecasting_features(raw_df, lat=cfg.location.latitude, lon=cfg.location.longitude)
    splits = get_chronological_splits(feat_df, cfg)

    eval_df = splits.test_df if split == "test" else (splits.val_df if split == "val" else splits.train_df)

    # 2. Load trained models & sensitivity matrix
    models_path = Path("results/models/forecast_models.joblib").resolve()
    scen_gen_path = Path("results/models/scenario_generator.joblib").resolve()
    sens_path = Path("results/models/sensitivity_model.json").resolve()

    forecast_models = joblib.load(models_path) if models_path.exists() else {}
    scenario_gen = joblib.load(scen_gen_path) if scen_gen_path.exists() else ScenarioGenerator(cfg)

    placement = determine_der_placement(cfg)

    if sens_path.exists():
        with open(sens_path, "r", encoding="utf-8") as f:
            sens_model = SensitivityModel.model_validate_json(f.read())
    else:
        from grid_edge_research.optimization.sensitivity_model import estimate_feeder_sensitivities
        sens_model = estimate_feeder_sensitivities(cfg, placement)

    # 3. Instantiate controller
    if cfg.controller == "rule_based":
        controller = RuleBasedController(cfg, placement)
    elif cfg.controller == "deterministic_mpc":
        controller = DeterministicMPCController(cfg, placement, sens_model)
    elif cfg.controller == "stochastic_mpc":
        controller = StochasticMPCController(cfg, placement, sens_model, scenario_gen)
    else:
        raise ValueError(f"Unknown controller '{cfg.controller}'")

    # 4. Run closed-loop simulation
    sim = ClosedLoopSimulator(
        config=cfg,
        controller=controller,
        placement=placement,
        scenario_generator=scenario_gen,
        forecast_models=forecast_models,
    )

    steps_to_run = n_steps if n_steps is not None else len(eval_df) - cfg.forecasting.forecast_horizon_hours
    records_df, metrics = sim.run_simulation(
        feature_df=eval_df,
        start_idx=0,
        n_steps=steps_to_run,
        experiment_id=exp_id,
        verbose=True,
    )

    # 5. Save results and resolved configuration
    records_df.to_parquet(run_dir / "step_records.parquet")
    records_df.to_csv(run_dir / "step_records.csv")

    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        f.write(metrics.model_dump_json(indent=2))

    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        f.write(cfg.model_dump_json(indent=2))

    logger.info(f"Completed experiment '{exp_id}'. Results saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single closed-loop grid experiment.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--controller", type=str, default=None, choices=["rule_based", "deterministic_mpc", "stochastic_mpc"])
    parser.add_argument("--pv-penetration", type=float, default=None, help="PV penetration float (e.g. 0.50)")
    parser.add_argument("--uncertainty", type=str, default=None, choices=["nominal", "moderate", "strong"])
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")

    args = parser.parse_args()
    run_experiment(
        config_path=args.config,
        controller_name=args.controller,
        pv_penetration=args.pv_penetration,
        uncertainty_level=args.uncertainty,
        n_steps=args.steps,
        split=args.split,
        run_id=args.run_id,
    )
