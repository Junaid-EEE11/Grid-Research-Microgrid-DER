"""
Publication-quality plotting module for IEEE Transactions-style figures.
All figures are styled with clean fonts, proper units, high DPI, and explicit legends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Matplotlib styling for academic publication
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    }
)


def plot_forecast_examples(
    load_true: pd.Series,
    load_preds: Dict[float, np.ndarray],
    pv_true: pd.Series,
    pv_preds: Dict[float, np.ndarray],
    timestamps: pd.DatetimeIndex,
    save_path: str | Path = "results/figures/forecast_load_pv_comparison.png",
) -> None:
    """Generate side-by-side time series plots with 80% prediction intervals."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Load Forecast Plot
    ax1 = axes[0]
    ax1.plot(timestamps, load_true, "k-", label="Realized Demand", linewidth=1.8)
    if 0.50 in load_preds:
        ax1.plot(timestamps, load_preds[0.50], "b--", label="Median Forecast ($q=0.50$)")
    if 0.10 in load_preds and 0.90 in load_preds:
        ax1.fill_between(
            timestamps,
            load_preds[0.10],
            load_preds[0.90],
            color="blue",
            alpha=0.20,
            label="80% Prediction Interval ($q_{0.10}-q_{0.90}$)",
        )
    ax1.set_ylabel("Demand (kW)")
    ax1.set_title("(a) Feeder Aggregate Active Demand Forecasting")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 2. PV Forecast Plot
    ax2 = axes[1]
    ax2.plot(timestamps, pv_true, "k-", label="Realized PV", linewidth=1.8)
    if 0.50 in pv_preds:
        ax2.plot(timestamps, pv_preds[0.50], "r--", label="Median Forecast ($q=0.50$)")
    if 0.10 in pv_preds and 0.90 in pv_preds:
        ax2.fill_between(
            timestamps,
            pv_preds[0.10],
            pv_preds[0.90],
            color="red",
            alpha=0.20,
            label="80% Prediction Interval ($q_{0.10}-q_{0.90}$)",
        )
    ax2.set_ylabel("PV Generation (p.u.)")
    ax2.set_xlabel("Time (UTC)")
    ax2.set_title("(b) Solar PV Generation Forecasting")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info(f"Saved forecast example figure to {out_p}")


def plot_calibration_reliability(
    nominal_quantiles: List[float],
    empirical_coverages: Dict[str, List[float]],
    save_path: str | Path = "results/figures/calibration_reliability_plot.png",
) -> None:
    """Generate probabilistic forecast reliability/calibration diagram."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "k--", label="Ideal Calibration ($y=x$)", linewidth=1.5)

    styles = {
        "well_calibrated": ("go-", "Well-Calibrated (Empirical Residuals)"),
        "under_dispersed": ("rs--", "Under-Dispersed (Overconfident)"),
        "over_dispersed": ("bd-.", "Over-Dispersed (Underconfident)"),
    }

    for key, (fmt, label) in styles.items():
        if key in empirical_coverages:
            ax.plot(nominal_quantiles, empirical_coverages[key], fmt, label=label, markersize=6)

    ax.set_xlabel("Nominal Quantile $\\tau$")
    ax.set_ylabel("Empirical Coverage Fraction")
    ax.set_title("Probabilistic Forecast Calibration Diagram")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info(f"Saved calibration reliability plot to {out_p}")


def plot_voltage_profile_comparison(
    records_dict: Dict[str, pd.DataFrame],
    save_path: str | Path = "results/figures/voltage_profile_comparison.png",
    hours_to_show: int = 48,
) -> None:
    """Plot physical minimum and maximum bus voltages across controllers over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    colors = {
        "rule_based": "#2ca02c",
        "deterministic_mpc": "#ff7f0e",
        "stochastic_mpc": "#1f77b4",
    }
    labels = {
        "rule_based": "Rule-Based Baseline",
        "deterministic_mpc": "Deterministic MPC",
        "stochastic_mpc": "Risk-Aware Stochastic MPC",
    }

    # Upper voltage plot
    for name, df in records_dict.items():
        sub = df.iloc[:hours_to_show]
        c = colors.get(name, "gray")
        lbl = labels.get(name, name)
        ax1.plot(sub.index, sub["v_max_pu"], label=lbl, color=c, linewidth=1.6)

    ax1.axhline(1.05, color="red", linestyle=":", label="ANSI Upper Limit (1.05 pu)", linewidth=1.8)
    ax1.set_ylabel("Max Voltage $V_{\\max}$ (p.u.)")
    ax1.set_title("(a) Feeder Maximum Bus Voltage Profile")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Lower voltage plot
    for name, df in records_dict.items():
        sub = df.iloc[:hours_to_show]
        c = colors.get(name, "gray")
        lbl = labels.get(name, name)
        ax2.plot(sub.index, sub["v_min_pu"], label=lbl, color=c, linewidth=1.6)

    ax2.axhline(0.95, color="red", linestyle=":", label="ANSI Lower Limit (0.95 pu)", linewidth=1.8)
    ax2.set_ylabel("Min Voltage $V_{\\min}$ (p.u.)")
    ax2.set_xlabel("Time")
    ax2.set_title("(b) Feeder Minimum Bus Voltage Profile")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info(f"Saved voltage profile comparison to {out_p}")


def plot_controller_matrix_summary(
    matrix_df: pd.DataFrame,
    save_pv_path: str | Path = "results/figures/controller_comparison_pv_penetration.png",
    save_unc_path: str | Path = "results/figures/controller_comparison_uncertainty.png",
) -> None:
    """Plot violation exposure across PV penetration levels and uncertainty stress levels."""
    # 1. PV Penetration Sweep Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    nominal_df = matrix_df[matrix_df["uncertainty_level"] == "nominal"]

    controllers = ["rule_based", "deterministic_mpc", "stochastic_mpc"]
    colors = {"rule_based": "#2ca02c", "deterministic_mpc": "#ff7f0e", "stochastic_mpc": "#1f77b4"}
    markers = {"rule_based": "s", "deterministic_mpc": "^", "stochastic_mpc": "o"}
    labels = {"rule_based": "Rule-Based Baseline", "deterministic_mpc": "Deterministic MPC", "stochastic_mpc": "Risk-Aware Stochastic MPC"}

    for c in controllers:
        sub = nominal_df[nominal_df["controller"] == c].sort_values("pv_penetration_pct")
        if not sub.empty:
            ax.plot(
                sub["pv_penetration_pct"],
                sub["total_voltage_violation_exposure"],
                color=colors[c],
                marker=markers[c],
                label=labels[c],
                linewidth=1.8,
                markersize=7,
            )

    ax.set_xlabel("PV Penetration Level (%)")
    ax.set_ylabel("Realized Voltage Violation Exposure (p.u.·h)")
    ax.set_title("Controller Voltage Reliability vs. PV Penetration")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_pv = Path(save_pv_path)
    out_pv.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pv, dpi=300)
    plt.close()

    # 2. Uncertainty Stress Sweep Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    pv50_df = matrix_df[matrix_df["pv_penetration_pct"] == 50.0]

    unc_order = ["nominal", "moderate", "strong"]
    for c in controllers:
        sub = pv50_df[pv50_df["controller"] == c].copy()
        if not sub.empty:
            sub["unc_rank"] = sub["uncertainty_level"].map({"nominal": 0, "moderate": 1, "strong": 2})
            sub = sub.sort_values("unc_rank")
            ax.plot(
                sub["uncertainty_level"],
                sub["total_voltage_violation_exposure"],
                color=colors[c],
                marker=markers[c],
                label=labels[c],
                linewidth=1.8,
                markersize=7,
            )

    ax.set_xlabel("Forecast Uncertainty Stress Level")
    ax.set_ylabel("Realized Voltage Violation Exposure (p.u.·h)")
    ax.set_title("Controller Robustness under Forecast Uncertainty Stress")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_unc = Path(save_unc_path)
    out_unc.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_unc, dpi=300)
    plt.close()
    logger.info("Saved matrix comparison plots.")


def plot_ablation_tradeoffs(
    ablation_df: pd.DataFrame,
    save_path: str | Path = "results/figures/ablation_tradeoff_pareto.png",
) -> None:
    """Plot reliability vs curtailment and battery throughput trade-offs for ablation variants."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in ablation_df.iterrows():
        name = row["controller"]
        x = row["total_pv_curtailment_kwh"]
        y = row["total_voltage_violation_exposure"]
        size = max(50.0, row["battery_throughput_kwh"] / 10.0)

        ax.scatter(x, y, s=size, alpha=0.7, edgecolors="black", label=name)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.set_xlabel("Total PV Curtailment (kWh)")
    ax.set_ylabel("Realized Voltage Violation Exposure (p.u.·h)")
    ax.set_title("Ablation Study: Reliability vs. Curtailment Trade-off")
    ax.grid(True)
    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info(f"Saved ablation trade-off figure to {out_p}")
