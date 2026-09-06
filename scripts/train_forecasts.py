"""
Script to train all baseline and probabilistic forecasting models,
evaluate on validation and test sets, and store models and residual pools.
"""

from __future__ import annotations

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
from grid_edge_research.data.loader import get_chronological_splits, load_processed_data
from grid_edge_research.forecasting.features import build_forecasting_features
from grid_edge_research.forecasting.metrics import compute_point_metrics, compute_probabilistic_metrics
from grid_edge_research.forecasting.models import PointGBRForecaster, QuantileGBRForecaster, SeasonalPersistenceForecaster
from grid_edge_research.uncertainty.scenario_generator import ScenarioGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_forecasting(
    config_path: str | None = None,
    output_dir: str | Path = "results",
) -> None:
    """Train persistence, point, and probabilistic forecast models and evaluate them."""
    cfg = load_config(config_path)
    out_path = Path(output_dir).resolve()
    model_dir = out_path / "models"
    table_dir = out_path / "tables"
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and create splits
    logger.info("Loading processed time-series data...")
    raw_df = load_processed_data(cfg)
    logger.info("Engineering causal forecasting features...")
    feat_df = build_forecasting_features(raw_df, lat=cfg.location.latitude, lon=cfg.location.longitude)

    splits = get_chronological_splits(feat_df, cfg)
    train_df = splits.train_df
    val_df = splits.val_df
    test_df = splits.test_df

    logger.info(
        f"Dataset split: Train={len(train_df)}h ({train_df.index[0]} to {train_df.index[-1]}), "
        f"Val={len(val_df)}h ({val_df.index[0]} to {val_df.index[-1]}), "
        f"Test={len(test_df)}h ({test_df.index[0]} to {test_df.index[-1]})"
    )

    models = {}
    metrics_records = []

    for target in ["load_kw", "pv_pu"]:
        logger.info(f"=== Training models for target: {target} ===")

        # 1. Persistence Baseline
        p_model = SeasonalPersistenceForecaster(target_col=target)
        p_model.fit(train_df)
        models[f"{target}_persistence"] = p_model

        for split_name, df_split in [("val", val_df), ("test", test_df)]:
            p_preds = p_model.predict(df_split)
            m = compute_point_metrics(df_split[target], p_preds, target_name=target, model_name="persistence", split_name=split_name)
            metrics_records.append(m.model_dump())

        # 2. Point GBR
        pt_model = PointGBRForecaster(
            target_col=target,
            n_estimators=cfg.forecasting.n_estimators,
            max_depth=cfg.forecasting.max_depth,
            learning_rate=cfg.forecasting.learning_rate,
            random_seed=cfg.random_seed,
        )
        pt_model.fit(train_df)
        models[f"{target}_point_gbr"] = pt_model

        for split_name, df_split in [("val", val_df), ("test", test_df)]:
            pt_preds = pt_model.predict(df_split)
            m = compute_point_metrics(df_split[target], pt_preds, target_name=target, model_name="point_gbr", split_name=split_name)
            metrics_records.append(m.model_dump())

        # 3. Quantile GBR
        q_model = QuantileGBRForecaster(
            target_col=target,
            quantiles=cfg.forecasting.quantiles,
            n_estimators=cfg.forecasting.n_estimators,
            max_depth=cfg.forecasting.max_depth,
            learning_rate=cfg.forecasting.learning_rate,
            random_seed=cfg.random_seed,
        )
        q_model.fit(train_df)
        models[f"{target}_quantile_gbr"] = q_model

        for split_name, df_split in [("val", val_df), ("test", test_df)]:
            q_preds = q_model.predict(df_split)
            m = compute_probabilistic_metrics(df_split[target], q_preds, target_name=target, model_name="quantile_gbr", split_name=split_name)
            metrics_records.append(m.model_dump())

    # 4. Save Models
    joblib.dump(models, model_dir / "forecast_models.joblib")
    logger.info(f"Saved trained forecast models to {model_dir / 'forecast_models.joblib'}")

    # 5. Extract and save Validation Residuals for Scenario Generator
    load_q_model = models["load_kw_quantile_gbr"]
    pv_q_model = models["pv_pu_quantile_gbr"]
    val_load_preds = load_q_model.predict(val_df)[0.50]
    val_pv_preds = pv_q_model.predict(val_df)[0.50]

    scenario_gen = ScenarioGenerator(cfg)
    scenario_gen.fit_residual_pool(
        load_true=val_df["load_kw"].values,
        load_pred_median=val_load_preds,
        pv_true=val_df["pv_pu"].values,
        pv_pred_median=val_pv_preds,
    )
    joblib.dump(scenario_gen, model_dir / "scenario_generator.joblib")
    logger.info(f"Saved scenario generator with validation residual pool to {model_dir / 'scenario_generator.joblib'}")

    # 6. Save metrics tables
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(table_dir / "forecast_metrics_summary.csv", index=False)
    metrics_df.to_json(table_dir / "forecast_metrics_summary.json", orient="records", indent=2)
    logger.info(f"Saved forecast evaluation summary to {table_dir / 'forecast_metrics_summary.csv'}")

    # Print summary
    print("\n" + "=" * 60)
    print("FORECAST EVALUATION SUMMARY (TEST SET)")
    print("=" * 60)
    test_metrics = metrics_df[metrics_df["split_name"] == "test"]
    print(test_metrics[["target_name", "model_name", "mae", "rmse", "nmae", "picp_80", "pinaw_80"]].to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_evaluate_forecasting()
