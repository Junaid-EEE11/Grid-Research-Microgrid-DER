"""
Comprehensive point and probabilistic forecast evaluation metrics.
Includes pinball loss, interval coverage (PICP), sharpness (PINAW), and calibration diagnostics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel


class ForecastMetrics(BaseModel):
    """Container for evaluation summary metrics."""
    target_name: str
    model_name: str
    split_name: str
    mae: float
    rmse: float
    nmae: float
    pinball_losses: Dict[str, float] = {}
    picp_80: float = 0.0         # Empirical coverage for [q0.10, q0.90] (nominal 0.80)
    pinaw_80: float = 0.0        # Mean interval width normalized
    coverage_error_80: float = 0.0


def compute_point_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_name: str = "load_kw",
    model_name: str = "point_gbr",
    split_name: str = "test",
) -> ForecastMetrics:
    """Compute standard point forecast metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    error = yt - yp
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.max(yt) - np.min(yt)) if (np.max(yt) - np.min(yt)) > 0 else float(np.mean(yt))
    nmae = float(mae / denom) if denom > 0 else 0.0

    return ForecastMetrics(
        target_name=target_name,
        model_name=model_name,
        split_name=split_name,
        mae=mae,
        rmse=rmse,
        nmae=nmae,
    )


def pinball_loss(
    y_true: np.ndarray,
    y_quantile_pred: np.ndarray,
    q: float,
) -> float:
    """
    Compute quantile pinball loss:
        L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    error = y_true - y_quantile_pred
    loss = np.maximum(q * error, (q - 1.0) * error)
    return float(np.mean(loss))


def compute_probabilistic_metrics(
    y_true: np.ndarray | pd.Series,
    quantile_preds: Dict[float, np.ndarray | pd.Series],
    target_name: str = "load_kw",
    model_name: str = "quantile_gbr",
    split_name: str = "test",
) -> ForecastMetrics:
    """
    Compute joint point and probabilistic evaluation metrics including pinball losses,
    empirical coverage, interval width, and calibration error.
    """
    yt = np.asarray(y_true, dtype=float)
    y_med = np.asarray(quantile_preds[0.5], dtype=float) if 0.5 in quantile_preds else np.asarray(list(quantile_preds.values())[0], dtype=float)

    base = compute_point_metrics(yt, y_med, target_name=target_name, model_name=model_name, split_name=split_name)

    pb_losses: Dict[str, float] = {}
    for q, yp in quantile_preds.items():
        yp_arr = np.asarray(yp, dtype=float)
        pb_losses[f"q_{int(round(q*100))}"] = pinball_loss(yt, yp_arr, q)

    base.pinball_losses = pb_losses

    # Interval metrics for 80% nominal interval [0.10, 0.90]
    if 0.10 in quantile_preds and 0.90 in quantile_preds:
        q_low = np.asarray(quantile_preds[0.10], dtype=float)
        q_high = np.asarray(quantile_preds[0.90], dtype=float)
        
        # Ensure quantile monotonic ordering
        q_high = np.maximum(q_high, q_low)

        covered = (yt >= q_low) & (yt <= q_high)
        picp = float(np.mean(covered))
        
        interval_width = q_high - q_low
        scale_range = float(np.max(yt) - np.min(yt)) if (np.max(yt) - np.min(yt)) > 0 else 1.0
        pinaw = float(np.mean(interval_width) / scale_range)
        cov_error = float(picp - 0.80)

        base.picp_80 = picp
        base.pinaw_80 = pinaw
        base.coverage_error_80 = cov_error

    return base
