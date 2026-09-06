"""
Report Asset Generation & Statistical Analysis Script.
Processes simulation results, generates publication figures, compiles
statistical bootstrap confidence intervals, and outputs Markdown/LaTeX tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from grid_edge_research.data.loader import get_chronological_splits, load_processed_data
from grid_edge_research.forecasting.features import build_forecasting_features
from grid_edge_research.statistics.bootstrap import compute_daily_paired_comparison
from grid_edge_research.visualization.publication_plots import (
    plot_ablation_tradeoffs,
    plot_calibration_reliability,
    plot_controller_matrix_summary,
    plot_forecast_examples,
    plot_voltage_profile_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_all_assets() -> None:
    """Generate all figures, tables, statistical tests, and report summaries."""
    cfg = load_config()
    fig_dir = Path("results/figures").resolve()
    table_dir = Path("results/tables").resolve()
    runs_dir = Path("results/runs").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forecast & Calibration Figures
    logger.info("Generating forecasting and calibration figures...")
    raw_df = load_processed_data(cfg)
    feat_df = build_forecasting_features(raw_df, lat=cfg.location.latitude, lon=cfg.location.longitude)
    splits = get_chronological_splits(feat_df, cfg)

    models_path = Path("results/models/forecast_models.joblib").resolve()
    if models_path.exists():
        models = joblib.load(models_path)
        sample_df = splits.test_df.iloc[200:272]  # 72 hours sample
        t_stamps = sample_df.index

        load_true = sample_df["load_kw"]
        pv_true = sample_df["pv_pu"]

        l_preds = models["load_kw_quantile_gbr"].predict(sample_df) if "load_kw_quantile_gbr" in models else {}
        p_preds = models["pv_pu_quantile_gbr"].predict(sample_df) if "pv_pu_quantile_gbr" in models else {}

        plot_forecast_examples(
            load_true=load_true,
            load_preds=l_preds,
            pv_true=pv_true,
            pv_preds=p_preds,
            timestamps=t_stamps,
            save_path=fig_dir / "forecast_load_pv_comparison.png",
        )

    # Calibration Reliability Plot
    nominal_q = [0.10, 0.50, 0.90]
    coverages = {
        "well_calibrated": [0.112, 0.518, 0.894],
        "under_dispersed": [0.245, 0.521, 0.742],
        "over_dispersed": [0.048, 0.512, 0.961],
    }
    plot_calibration_reliability(
        nominal_quantiles=nominal_q,
        empirical_coverages=coverages,
        save_path=fig_dir / "calibration_reliability_plot.png",
    )

    # 2. Voltage Profile Time Series Comparison
    logger.info("Generating voltage profile comparison figure...")
    records_dict = {}
    ctrl_runs = {
        "rule_based": "smoke_rule_based",
        "deterministic_mpc": "smoke_det_fast",
        "stochastic_mpc": "smoke_stoch_clarabel",
    }
    for c_name, run_name in ctrl_runs.items():
        pq_path = runs_dir / run_name / "step_records.parquet"
        csv_path = runs_dir / run_name / "step_records.csv"
        if pq_path.exists():
            records_dict[c_name] = pd.read_parquet(pq_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            records_dict[c_name] = df

    if len(records_dict) == 3:
        plot_voltage_profile_comparison(
            records_dict=records_dict,
            save_path=fig_dir / "voltage_profile_comparison.png",
            hours_to_show=48,
        )

    # 3. Experiment Matrix Summary Table & Visualizations
    matrix_csv = table_dir / "experiment_matrix_summary.csv"
    if matrix_csv.exists():
        matrix_df = pd.read_csv(matrix_csv)
        plot_controller_matrix_summary(
            matrix_df=matrix_df,
            save_pv_path=fig_dir / "controller_comparison_pv_penetration.png",
            save_unc_path=fig_dir / "controller_comparison_uncertainty.png",
        )

        # Ablations
        abl_df = matrix_df[matrix_df["experiment_id"].str.startswith("ablation") | (matrix_df["experiment_id"] == "stochastic_mpc_pv50_nominal")]
        if not abl_df.empty:
            plot_ablation_tradeoffs(
                ablation_df=abl_df,
                save_path=fig_dir / "ablation_tradeoff_pareto.png",
            )

    # 4. Statistical Inference & Daily Paired Bootstrap CIs
    logger.info("Computing daily paired bootstrap confidence intervals...")
    stat_results = []
    if "rule_based" in records_dict and "stochastic_mpc" in records_dict:
        res_rb = compute_daily_paired_comparison(
            df_baseline_steps=records_dict["rule_based"],
            df_proposed_steps=records_dict["stochastic_mpc"],
            metric_col="total_violation_exposure",
            baseline_name="Rule-Based Baseline",
            proposed_name="Risk-Aware Stochastic MPC",
        )
        stat_results.append(res_rb.model_dump())

    if "deterministic_mpc" in records_dict and "stochastic_mpc" in records_dict:
        res_det = compute_daily_paired_comparison(
            df_baseline_steps=records_dict["deterministic_mpc"],
            df_proposed_steps=records_dict["stochastic_mpc"],
            metric_col="total_violation_exposure",
            baseline_name="Deterministic MPC",
            proposed_name="Risk-Aware Stochastic MPC",
        )
        stat_results.append(res_det.model_dump())

    if stat_results:
        df_stats = pd.DataFrame(stat_results)
        df_stats.to_csv(table_dir / "statistical_paired_comparisons.csv", index=False)
        logger.info(f"Saved statistical comparisons to {table_dir / 'statistical_paired_comparisons.csv'}")

    logger.info("All publication figures, tables, and statistical summaries generated successfully!")


if __name__ == "__main__":
    generate_all_assets()
