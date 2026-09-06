"""
Script to merge and validate aligned hourly load and PV profiles,
producing the canonical time-series dataset for training, validation, and testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from scripts.fetch_feeder import fetch_feeder_files
from scripts.fetch_load_data import fetch_and_process_load_data
from scripts.fetch_weather_data import fetch_and_process_weather_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_aligned_dataset(
    processed_dir: Path | str = "data/processed",
    force: bool = False,
) -> Path:
    """Merge load and PV profiles, validate physical constraints, and save aligned dataset."""
    p_dir = Path(processed_dir).resolve()
    p_dir.mkdir(parents=True, exist_ok=True)

    target_parquet = p_dir / "feeder_timeseries_hourly.parquet"
    target_csv = p_dir / "feeder_timeseries_hourly.csv"
    manifest_file = p_dir / "dataset_manifest.json"

    if target_parquet.exists() and not force:
        logger.info(f"Aligned dataset already exists at {target_parquet}")
        return target_parquet

    # 1. Ensure feeder, load, and weather data are present
    load_path = p_dir / "feeder_load_hourly.parquet"
    pv_path = p_dir / "feeder_pv_hourly.parquet"

    if not load_path.exists():
        logger.info("Fetching and processing load data...")
        fetch_and_process_load_data(processed_dir=p_dir)

    if not pv_path.exists():
        logger.info("Fetching and processing solar/weather data...")
        fetch_and_process_weather_data(processed_dir=p_dir)

    # 2. Load both series
    load_df = pd.read_parquet(load_path)
    pv_df = pd.read_parquet(pv_path)

    # Align timestamps
    common_index = load_df.index.intersection(pv_df.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps between load and PV datasets!")

    merged_df = pd.DataFrame(index=common_index)
    merged_df["load_kw"] = load_df.loc[common_index, "load_kw"]
    merged_df["load_pu"] = load_df.loc[common_index, "load_pu"]
    merged_df["pv_pu"] = pv_df.loc[common_index, "pv_pu"]
    if "ghi" in pv_df.columns:
        merged_df["ghi"] = pv_df.loc[common_index, "ghi"]
    if "temp_c" in pv_df.columns:
        merged_df["temp_c"] = pv_df.loc[common_index, "temp_c"]

    # Physical Assertions
    assert (merged_df["load_kw"] >= 0).all(), "Load power must be non-negative!"
    assert (merged_df["pv_pu"] >= -1e-5).all(), "PV output must be non-negative!"
    assert (merged_df["pv_pu"] <= 1.0 + 1e-5).all(), "PV per-unit output must not exceed 1.0!"
    merged_df["pv_pu"] = merged_df["pv_pu"].clip(0.0, 1.0)

    # Check for null values
    if merged_df.isna().sum().sum() > 0:
        logger.warning("Interpolating remaining NaN values in merged dataset...")
        merged_df = merged_df.interpolate(method="time").bfill().ffill()

    # Save to Parquet and CSV
    merged_df.to_parquet(target_parquet)
    merged_df.to_csv(target_csv)
    logger.info(f"Saved aligned time series dataset ({len(merged_df)} hours) to {target_parquet}")

    # Write dataset manifest
    manifest = {
        "dataset_name": "Aligned IEEE 123 Feeder Load & PV Timeseries",
        "total_hours": len(merged_df),
        "start_time": str(merged_df.index.min()),
        "end_time": str(merged_df.index.max()),
        "columns": list(merged_df.columns),
        "load_kw_mean": float(merged_df["load_kw"].mean()),
        "load_kw_max": float(merged_df["load_kw"].max()),
        "pv_pu_mean": float(merged_df["pv_pu"].mean()),
        "pv_pu_max": float(merged_df["pv_pu"].max()),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return target_parquet


if __name__ == "__main__":
    prepare_aligned_dataset()
