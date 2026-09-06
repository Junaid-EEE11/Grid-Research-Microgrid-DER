"""
Data loading utilities for raw and processed time-series and feeder files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from pydantic import BaseModel

from grid_edge_research.config import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


class DatasetSplits(BaseModel):
    """Container for chronological train, validation, and test datasets."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: dict

    class Config:
        arbitrary_types_allowed = True


def load_processed_data(
    config: Optional[ExperimentConfig] = None,
    processed_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Load merged hourly load and PV dataset.
    
    Returns:
        pd.DataFrame with DatetimeIndex and columns ['load_kw', 'pv_pu', 'load_pu', 'ghi', 'temp_c'].
    """
    if config is None:
        config = load_config()

    p_dir = Path(processed_dir) if processed_dir else config.paths.processed_data_dir
    parquet_path = p_dir / "feeder_timeseries_hourly.parquet"
    csv_path = p_dir / "feeder_timeseries_hourly.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        raise FileNotFoundError(
            f"Processed time-series dataset not found in {p_dir}. "
            "Please run 'make data' or 'python scripts/prepare_data.py'."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df


def get_chronological_splits(
    df: pd.DataFrame,
    config: Optional[ExperimentConfig] = None,
) -> DatasetSplits:
    """
    Perform strict chronological split into Train, Validation, and Test sets
    without shuffling or data leakage.
    """
    if config is None:
        config = load_config()

    dates = config.date_ranges
    train_start = pd.Timestamp(dates.train_start)
    train_end = pd.Timestamp(dates.train_end)
    val_start = pd.Timestamp(dates.val_start)
    val_end = pd.Timestamp(dates.val_end)
    test_start = pd.Timestamp(dates.test_start)
    test_end = pd.Timestamp(dates.test_end)

    # Ensure chronological validity
    if not (train_start < train_end < val_start < val_end < test_start < test_end):
        raise ValueError(
            f"Date ranges must be strictly chronological: "
            f"Train [{train_start} to {train_end}], "
            f"Val [{val_start} to {val_end}], "
            f"Test [{test_start} to {test_end}]"
        )

    train_df = df.loc[train_start:train_end].copy()
    val_df = df.loc[val_start:val_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"One or more splits are empty! Train rows: {len(train_df)}, "
            f"Val rows: {len(val_df)}, Test rows: {len(test_df)}"
        )

    metadata = {
        "train_range": [str(train_start), str(train_end)],
        "val_range": [str(val_start), str(val_end)],
        "test_range": [str(test_start), str(test_end)],
        "train_hours": len(train_df),
        "val_hours": len(val_df),
        "test_hours": len(test_df),
    }

    return DatasetSplits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        metadata=metadata,
    )
