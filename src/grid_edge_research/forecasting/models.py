"""
Forecasting models for multi-step load and solar PV prediction:
1. Seasonal Persistence Baseline
2. Point Gradient Boosting Regressor (GBR)
3. Quantile Gradient Boosting Regressor (q=0.10, 0.50, 0.90) with Monotonicity Enforcement
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from grid_edge_research.config import ExperimentConfig, load_config

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "solar_zenith", "solar_elevation", "cos_zenith", "clearsky_ghi",
    "load_lag_1", "load_lag_2", "load_lag_24", "load_lag_48", "load_lag_168",
    "load_roll_mean_24", "load_roll_std_24", "load_roll_mean_168",
    "pv_lag_1", "pv_lag_2", "pv_lag_24", "pv_roll_mean_24",
]


class BaseForecaster:
    """Abstract forecaster interface."""

    def fit(self, train_df: pd.DataFrame) -> BaseForecaster:
        raise NotImplementedError

    def predict(self, feature_df: pd.DataFrame) -> Union[np.ndarray, Dict[float, np.ndarray]]:
        raise NotImplementedError


class SeasonalPersistenceForecaster(BaseForecaster):
    """
    Seasonal persistence baseline.
    Load: Same hour of previous week (lag 168h) or day (lag 24h).
    PV: Clearsky shape scaled by previous day's ratio, forced zero at night.
    """

    def __init__(self, target_col: str = "load_kw"):
        self.target_col = target_col

    def fit(self, train_df: pd.DataFrame) -> SeasonalPersistenceForecaster:
        return self

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        if "load" in self.target_col:
            if "load_lag_168" in feature_df.columns:
                preds = feature_df["load_lag_168"].values.copy()
            else:
                preds = feature_df["load_lag_24"].values.copy()
            return np.clip(preds, 0.0, None)
        else:
            preds = feature_df["pv_lag_24"].values.copy() if "pv_lag_24" in feature_df.columns else np.zeros(len(feature_df))
            if "solar_zenith" in feature_df.columns:
                night = feature_df["solar_zenith"].values >= 90.0
                preds[night] = 0.0
            return np.clip(preds, 0.0, 1.0)


class PointGBRForecaster(BaseForecaster):
    """Point Gradient Boosting Forecaster for conditional mean."""

    def __init__(
        self,
        target_col: str = "load_kw",
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_seed: int = 42,
    ):
        self.target_col = target_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
        )

    def fit(self, train_df: pd.DataFrame) -> PointGBRForecaster:
        cols = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[cols].values
        y = train_df[self.target_col].values
        self.model.fit(X, y)
        return self

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in FEATURE_COLS if c in feature_df.columns]
        X = feature_df[cols].values
        preds = self.model.predict(X)
        if "pv" in self.target_col:
            preds = np.clip(preds, 0.0, 1.0)
            if "solar_zenith" in feature_df.columns:
                night = feature_df["solar_zenith"].values >= 90.0
                preds[night] = 0.0
        else:
            preds = np.clip(preds, 0.0, None)
        return preds


class QuantileGBRForecaster(BaseForecaster):
    """
    Probabilistic Quantile Gradient Boosting Forecaster for quantiles q in [0.10, 0.50, 0.90].
    Enforces monotonicity (non-crossing quantiles) and physical bounds.
    """

    def __init__(
        self,
        target_col: str = "load_kw",
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_seed: int = 42,
    ):
        self.target_col = target_col
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.models: Dict[float, GradientBoostingRegressor] = {}

    def fit(self, train_df: pd.DataFrame) -> QuantileGBRForecaster:
        cols = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[cols].values
        y = train_df[self.target_col].values

        logger.info(f"Training Quantile GBR for {self.target_col} across quantiles {self.quantiles}...")
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_seed,
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict(self, feature_df: pd.DataFrame) -> Dict[float, np.ndarray]:
        cols = [c for c in FEATURE_COLS if c in feature_df.columns]
        X = feature_df[cols].values
        preds: Dict[float, np.ndarray] = {}

        for q in sorted(self.quantiles):
            yp = self.models[q].predict(X)
            preds[q] = yp

        # Non-crossing Quantile Monotonicity Enforcement (Isotonic sorting across quantiles)
        sorted_qs = sorted(self.quantiles)
        pred_matrix = np.column_stack([preds[q] for q in sorted_qs])
        pred_matrix = np.sort(pred_matrix, axis=1)

        for i, q in enumerate(sorted_qs):
            col_pred = pred_matrix[:, i]
            if "pv" in self.target_col:
                col_pred = np.clip(col_pred, 0.0, 1.0)
                if "solar_zenith" in feature_df.columns:
                    night = feature_df["solar_zenith"].values >= 90.0
                    col_pred[night] = 0.0
            else:
                col_pred = np.clip(col_pred, 0.0, None)
            preds[q] = col_pred

        return preds
