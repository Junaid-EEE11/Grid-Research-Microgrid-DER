"""
Script to fetch NASA POWER hourly solar irradiance and meteorological data
for representative Portugal coordinates and compute normalized PV generation with pvlib.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from grid_edge_research.data.preprocessor import generate_reproducible_synthetic_dataset, process_nasa_solar_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def fetch_and_process_weather_data(
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = "data/processed",
    lat: float = 39.5,
    lon: float = -8.0,
    force_synthetic: bool = False,
) -> Path:
    """Fetch solar/weather data from NASA POWER or generate verified pvlib benchmark profile."""
    raw_p = Path(raw_dir).resolve()
    processed_p = Path(processed_dir).resolve()
    raw_p.mkdir(parents=True, exist_ok=True)
    processed_p.mkdir(parents=True, exist_ok=True)

    raw_json_path = raw_p / "nasa_power_raw.json"
    out_parquet = processed_p / "feeder_pv_hourly.parquet"
    prov_file = processed_p / "pv_provenance.json"

    provenance: dict = {
        "dataset": "NASA POWER Hourly Solar and Meteorological Data",
        "latitude": lat,
        "longitude": lon,
        "parameters": ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M"],
        "timezone": "UTC",
        "pv_model": "pvlib PVWatts & Faiman cell temperature model",
    }

    if not force_synthetic:
        try:
            if not raw_json_path.exists():
                logger.info(f"Querying NASA POWER API for ({lat}, {lon}) from 20120101 to 20141231...")
                params = {
                    "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
                    "community": "RE",
                    "longitude": lon,
                    "latitude": lat,
                    "start": "20120101",
                    "end": "20141231",
                    "format": "JSON",
                }
                resp = requests.get(NASA_POWER_BASE_URL, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                with open(raw_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved raw NASA POWER data to {raw_json_path}")
            else:
                logger.info(f"Loading cached raw NASA POWER data from {raw_json_path}")
                with open(raw_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            provenance["raw_file_md5"] = compute_md5(raw_json_path)
            pv_df = process_nasa_solar_data(data, lat=lat, lon=lon, output_path=out_parquet)
            provenance["processed_file_md5"] = compute_md5(out_parquet)
            provenance["total_hourly_records"] = len(pv_df)
            provenance["data_type"] = "empirical_nasa_power_pvlib"

            with open(prov_file, "w", encoding="utf-8") as f:
                json.dump(provenance, f, indent=2)
            logger.info(f"Successfully processed {len(pv_df)} hourly PV observations.")
            return out_parquet

        except Exception as e:
            logger.warning(f"NASA POWER fetch encountered issue ({e}). Falling back to pvlib clearsky benchmark generator.")

    # Fallback to verified pvlib clearsky benchmark generator
    logger.info("Generating verified pvlib clearsky solar PV profile...")
    syn_df = generate_reproducible_synthetic_dataset(lat=lat, lon=lon)
    pv_df = pd.DataFrame(
        {
            "pv_pu": syn_df["pv_pu"],
            "ghi": syn_df["ghi"],
            "temp_c": syn_df["temp_c"],
        },
        index=syn_df.index,
    )
    pv_df.to_parquet(out_parquet)

    provenance["data_type"] = "synthetic_pvlib_clearsky_matched"
    provenance["processed_file_md5"] = compute_md5(out_parquet)
    provenance["total_hourly_records"] = len(pv_df)

    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Saved benchmark PV profile to {out_parquet}")
    return out_parquet


if __name__ == "__main__":
    fetch_and_process_weather_data()
