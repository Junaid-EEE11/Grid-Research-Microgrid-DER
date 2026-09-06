"""
Data preprocessing, feature transformation, solar physical modeling via pvlib,
and normalization for aggregate load and PV generation profiles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pvlib

from grid_edge_research.config import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


def process_uci_load_data(
    raw_csv_or_zip: Path | str,
    output_path: Optional[Path | str] = None,
    min_active_ratio: float = 0.90,
) -> pd.DataFrame:
    """
    Process raw UCI ElectricityLoadDiagrams20112014 dataset.
    
    1. Parse 15-minute intervals (format: YYYY-MM-DD HH:MM:SS; kW with ',' decimal).
    2. Filter out clients with excess zeros or inactive periods before 2012.
    3. Aggregate across active Portuguese clients.
    4. Resample to 1-hour average demand (kW).
    5. Clean timestamps and missing values via linear interpolation.
    """
    p = Path(raw_csv_or_zip)
    logger.info(f"Processing raw UCI load data from {p}...")

    df = pd.read_csv(
        p,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        low_memory=False,
    )
    df.index.name = "timestamp"

    sub_df = df.loc["2012-01-01":"2014-12-31"]
    active_mask = (sub_df > 0).mean(axis=0) >= min_active_ratio
    active_clients = sub_df.columns[active_mask].tolist()
    logger.info(f"Selected {len(active_clients)} of {len(df.columns)} active clients meeting threshold.")

    aggregate_15min_kw = sub_df[active_clients].sum(axis=1)

    hourly_kw = aggregate_15min_kw.resample("1h").mean()
    hourly_kw = hourly_kw.interpolate(method="time").bfill().ffill()

    res_df = pd.DataFrame({"load_kw": hourly_kw}, index=hourly_kw.index)
    res_df["load_pu"] = res_df["load_kw"] / res_df["load_kw"].max()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_parquet(out_p)
        logger.info(f"Saved processed hourly load to {out_p}")

    return res_df


def process_nasa_solar_data(
    raw_json_or_df: dict | pd.DataFrame | Path | str,
    lat: float = 39.5,
    lon: float = -8.0,
    surface_tilt: float = 35.0,
    surface_azimuth: float = 180.0,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Convert NASA POWER solar irradiance and temperature time series to normalized PV generation profile
    using pvlib physical transposition and PVWatts model.
    """
    logger.info(f"Processing solar/weather data with pvlib for ({lat}, {lon})...")

    if isinstance(raw_json_or_df, (Path, str)):
        p = Path(raw_json_or_df)
        if p.suffix == ".json":
            import json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = data["properties"]["parameter"]
            ghi_dict = params.get("ALLSKY_SFC_SW_DWN", {})
            temp_dict = params.get("T2M", {})
            timestamps = [pd.to_datetime(k, format="%Y%m%d%H") for k in ghi_dict.keys()]
            ghi = list(ghi_dict.values())
            temp_c = [temp_dict.get(k, 20.0) for k in ghi_dict.keys()]
            df = pd.DataFrame({"ghi": ghi, "temp_c": temp_c}, index=pd.DatetimeIndex(timestamps, tz="UTC"))
        else:
            df = pd.read_csv(p, parse_dates=True, index_col=0)
    elif isinstance(raw_json_or_df, dict):
        params = raw_json_or_df["properties"]["parameter"]
        ghi_dict = params.get("ALLSKY_SFC_SW_DWN", {})
        temp_dict = params.get("T2M", {})
        timestamps = [pd.to_datetime(k, format="%Y%m%d%H") for k in ghi_dict.keys()]
        ghi = list(ghi_dict.values())
        temp_c = [temp_dict.get(k, 20.0) for k in ghi_dict.keys()]
        df = pd.DataFrame({"ghi": ghi, "temp_c": temp_c}, index=pd.DatetimeIndex(timestamps, tz="UTC"))
    else:
        df = raw_json_or_df.copy()

    # Ensure tz-aware UTC DatetimeIndex
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Filter out missing NASA flags (-999.0)
    df["ghi"] = df["ghi"].replace(-999.0, np.nan).interpolate(method="time").fillna(0.0)
    df["temp_c"] = df["temp_c"].replace(-999.0, np.nan).interpolate(method="time").fillna(20.0)

    # Compute solar position
    times = df.index
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # Estimate DNI and DHI using Erbs model if only GHI is present
    erbs = pvlib.irradiance.erbs(df["ghi"], solpos["zenith"], times)
    dni = erbs["dni"].fillna(0.0)
    dhi = erbs["dhi"].fillna(0.0)

    # Calculate plane-of-array (POA) irradiance with dni_extra passed
    poa_sky = pvlib.irradiance.get_sky_diffuse(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dhi=dhi,
        dni=dni,
        ghi=df["ghi"],
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni_extra=dni_extra,
        model="haydavies",
    )
    poa_ground = pvlib.irradiance.get_ground_diffuse(surface_tilt, df["ghi"])
    poa_direct = pvlib.irradiance.beam_component(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
    )
    poa_global = poa_direct + poa_sky + poa_ground
    poa_global = poa_global.clip(lower=0.0)

    # Cell temperature modeling (Faiman model)
    cell_temp = pvlib.temperature.faiman(poa_global, df["temp_c"], wind_speed=2.0)

    # PVWatts DC and AC power computation
    p_dc = pvlib.pvsystem.pvwatts_dc(poa_global, cell_temp, pdc0=1000.0, gamma_pdc=-0.004)
    p_ac = pvlib.inverter.pvwatts(p_dc, pdc0=1000.0, eta_inv_nom=0.96)
    p_ac = p_ac.fillna(0.0).clip(lower=0.0)

    # Nighttime force zero where zenith >= 90 deg
    night_mask = solpos["zenith"] >= 90.0
    p_ac[night_mask] = 0.0

    # Normalize to [0, 1] per unit
    p_pv_pu = p_ac / 1000.0
    p_pv_pu = p_pv_pu.clip(0.0, 1.0)

    out_df = pd.DataFrame(
        {
            "ghi": df["ghi"].values,
            "temp_c": df["temp_c"].values,
            "poa_global": poa_global.values,
            "pv_pu": p_pv_pu.values,
        },
        index=df.index.tz_localize(None),  # store as tz-naive UTC
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_p)
        logger.info(f"Saved processed PV time series to {out_p}")

    return out_df


def generate_reproducible_synthetic_dataset(
    start_date: str = "2012-01-01 00:00:00",
    end_date: str = "2014-12-31 23:00:00",
    seed: int = 42,
    lat: float = 39.5,
    lon: float = -8.0,
) -> pd.DataFrame:
    """
    Generate authentic physics-based synthetic benchmark dataset for offline testing
    using pvlib clearsky solar model and realistic diurnal/weekly/seasonal demand curves.
    """
    logger.info(f"Generating synthetic benchmark dataset from {start_date} to {end_date}...")
    rng = np.random.RandomState(seed)
    times = pd.date_range(start_date, end_date, freq="1h")
    n = len(times)

    # 1. Demand Profile Generation
    hour = times.hour.values
    dayofweek = times.dayofweek.values
    dayofyear = times.dayofyear.values

    diurnal = 0.3 * np.sin(2 * np.pi * (hour - 6) / 24) + 0.15 * np.sin(4 * np.pi * (hour - 6) / 24)
    weekly = np.where(dayofweek < 5, 0.15, -0.10)
    seasonal = 0.20 * np.cos(2 * np.pi * (dayofyear - 15) / 365)
    noise = rng.normal(0, 0.05, size=n)

    nominal_peak_kw = 3490.0
    load_pu = 0.60 + diurnal + weekly + seasonal + noise
    load_pu = np.clip(load_pu, 0.20, 1.05)
    load_kw = load_pu * nominal_peak_kw

    # 2. Solar PV Profile Generation using pvlib Ineichen clearsky model
    tz_times = times.tz_localize("UTC")
    location = pvlib.location.Location(lat, lon, tz="UTC", altitude=150)
    clearsky = location.get_clearsky(tz_times, model="ineichen")
    ghi_clear = clearsky["ghi"].values

    cloud_state = 1.0
    cloud_factors = np.zeros(n)
    for i in range(n):
        if hour[i] == 0:
            daily_weather = rng.choice([1.0, 0.8, 0.5, 0.2], p=[0.5, 0.25, 0.15, 0.10])
        step_noise = rng.normal(0, 0.08)
        cloud_state = 0.85 * cloud_state + 0.15 * daily_weather + step_noise
        cloud_factors[i] = np.clip(cloud_state, 0.05, 1.0)

    ghi = ghi_clear * cloud_factors
    temp_c = 15.0 + 8.0 * np.sin(2 * np.pi * (hour - 9) / 24) + 10.0 * np.cos(2 * np.pi * (dayofyear - 190) / 365)

    raw_solar_df = pd.DataFrame({"ghi": ghi, "temp_c": temp_c}, index=times)
    solar_processed = process_nasa_solar_data(raw_solar_df, lat=lat, lon=lon)

    df = pd.DataFrame(
        {
            "load_kw": load_kw,
            "load_pu": load_pu / load_pu.max(),
            "pv_pu": solar_processed["pv_pu"].values,
            "ghi": ghi,
            "temp_c": temp_c,
        },
        index=times,
    )
    df.index.name = "timestamp"
    return df
