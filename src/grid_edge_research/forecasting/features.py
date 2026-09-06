"""
Feature engineering and lag matrix construction for multi-step load and PV forecasting.
Ensures strict chronological alignment with zero future leakage.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger(__name__)


def build_forecasting_features(
    df: pd.DataFrame,
    lat: float = 39.5,
    lon: float = -8.0,
) -> pd.DataFrame:
    """
    Build rich calendar, cyclical, lag, rolling, and solar physical features.
    
    All lags and rolling windows strictly reference past time steps (t-k, k >= 1)
    to prevent information leakage during forecasting.
    """
    feat_df = df.copy()

    # 1. Calendar & Cyclical Features
    hour = feat_df.index.hour
    dayofweek = feat_df.index.dayofweek
    dayofyear = feat_df.index.dayofyear
    month = feat_df.index.month

    feat_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat_df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    feat_df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    feat_df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    feat_df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    feat_df["is_weekend"] = (dayofweek >= 5).astype(float)

    # 2. Solar Physical Clearsky Features
    tz_index = feat_df.index.tz_localize("UTC") if feat_df.index.tz is None else feat_df.index
    solpos = pvlib.solarposition.get_solarposition(tz_index, lat, lon)
    feat_df["solar_zenith"] = solpos["zenith"].values
    feat_df["solar_elevation"] = solpos["elevation"].values
    feat_df["cos_zenith"] = np.maximum(0.0, np.cos(np.radians(solpos["zenith"].values)))

    # Clearsky GHI reference
    location = pvlib.location.Location(lat, lon, tz="UTC", altitude=150)
    clearsky = location.get_clearsky(tz_index, model="ineichen")
    feat_df["clearsky_ghi"] = clearsky["ghi"].values

    # 3. Lagged Demand Features
    for lag in [1, 2, 3, 24, 48, 168]:
        feat_df[f"load_lag_{lag}"] = feat_df["load_kw"].shift(lag)
        feat_df[f"load_pu_lag_{lag}"] = feat_df["load_pu"].shift(lag)

    # Rolling demand statistics over past 24h and 168h (shifted by 1 to exclude current t)
    feat_df["load_roll_mean_24"] = feat_df["load_kw"].shift(1).rolling(24, min_periods=1).mean()
    feat_df["load_roll_std_24"] = feat_df["load_kw"].shift(1).rolling(24, min_periods=1).std().fillna(0.0)
    feat_df["load_roll_mean_168"] = feat_df["load_kw"].shift(1).rolling(168, min_periods=1).mean()

    # 4. Lagged PV Features
    for lag in [1, 2, 24, 48]:
        feat_df[f"pv_lag_{lag}"] = feat_df["pv_pu"].shift(lag)
    
    feat_df["pv_roll_mean_24"] = feat_df["pv_pu"].shift(1).rolling(24, min_periods=1).mean()

    # Weather lags if present
    if "temp_c" in feat_df.columns:
        feat_df["temp_lag_1"] = feat_df["temp_c"].shift(1)
        feat_df["temp_lag_24"] = feat_df["temp_c"].shift(24)

    # Drop initial rows with unfillable lags (up to 168 hours = 1 week)
    feat_df = feat_df.bfill()
    return feat_df
