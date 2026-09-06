"""
Unit tests for forecasting models, causal lag features, and probabilistic evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest

from grid_edge_research.forecasting.metrics import compute_probabilistic_metrics
from grid_edge_research.forecasting.models import QuantileGBRForecaster, FEATURE_COLS


def test_quantile_monotonicity():
    """Verify that quantile predictions maintain strict non-crossing order q_10 <= q_50 <= q_90."""
    quantiles = [0.10, 0.50, 0.90]
    forecaster = QuantileGBRForecaster(target_col="load_kw", quantiles=quantiles, n_estimators=10)

    # Synthetic training data
    np.random.seed(42)
    data = {col: np.random.randn(100) for col in FEATURE_COLS}
    data["load_kw"] = np.random.uniform(10, 100, 100)
    train_df = pd.DataFrame(data)

    forecaster.fit(train_df)

    test_data = {col: np.random.randn(20) for col in FEATURE_COLS}
    test_df = pd.DataFrame(test_data)
    preds = forecaster.predict(test_df)

    assert 0.10 in preds and 0.50 in preds and 0.90 in preds
    q10 = preds[0.10]
    q50 = preds[0.50]
    q90 = preds[0.90]

    assert np.all(q10 <= q50 + 1e-6), "q_10 must not exceed q_50"
    assert np.all(q50 <= q90 + 1e-6), "q_50 must not exceed q_90"


def test_pinball_loss_and_coverage_metric():
    """Verify pinball loss and empirical coverage metrics on known synthetic values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_point = np.array([12.0, 18.0, 32.0, 38.0])

    q_dict = {
        0.10: np.array([5.0, 15.0, 25.0, 35.0]),
        0.50: y_pred_point,
        0.90: np.array([15.0, 25.0, 35.0, 45.0]),
    }

    metrics = compute_probabilistic_metrics(y_true, q_dict)

    # MAE = (|2| + |2| + |2| + |2|) / 4 = 2.0
    assert np.isclose(metrics.mae, 2.0)
    assert np.isclose(metrics.rmse, 2.0)

    # All true points lie strictly inside [q_0.1, q_0.9] -> PICP 80% should be 1.0
    assert metrics.picp_80 == 1.0
