"""
Multi-step forecast uncertainty scenario generation via block-bootstrap validation residuals,
quantile reconstruction, and controlled calibration inflation/deflation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel

from grid_edge_research.config import ExperimentConfig, UncertaintyConfig, load_config

logger = logging.getLogger(__name__)


class ScenarioSet(BaseModel):
    """Container for multi-step trajectory scenarios across the MPC horizon."""
    horizon_hours: int
    n_scenarios: int
    timestamps: List[str]
    load_scenarios_kw: List[List[float]]   # [n_scenarios, horizon]
    load_scenarios_pu: List[List[float]]
    pv_scenarios_pu: List[List[float]]
    load_median_kw: List[float]
    load_median_pu: List[float]
    pv_median_pu: List[float]
    calibration_mode: str
    inflation_factor: float
    seed: int


class ScenarioGenerator:
    """
    Generates physically valid multi-step load and PV uncertainty trajectories
    using validation-set block-bootstrap residuals.
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        load_val_residuals: Optional[np.ndarray] = None,
        pv_val_residuals: Optional[np.ndarray] = None,
    ):
        self.config = config or load_config()
        self.unc_cfg = self.config.uncertainty
        self.rng = np.random.RandomState(self.unc_cfg.random_seed)

        # Store or initialize residual pools (24-hour blocks)
        self.load_residuals = load_val_residuals if load_val_residuals is not None else np.zeros((100, 24))
        self.pv_residuals = pv_val_residuals if pv_val_residuals is not None else np.zeros((100, 24))

    def fit_residual_pool(
        self,
        load_true: np.ndarray,
        load_pred_median: np.ndarray,
        pv_true: np.ndarray,
        pv_pred_median: np.ndarray,
        block_len: int = 24,
    ) -> ScenarioGenerator:
        """Extract multi-step residual trajectories from validation evaluation."""
        n = min(len(load_true), len(load_pred_median), len(pv_true), len(pv_pred_median))
        n_blocks = n // block_len

        if n_blocks == 0:
            logger.warning("Not enough samples to extract full 24h residual blocks.")
            return self

        load_res_blocks = []
        pv_res_blocks = []

        for i in range(n_blocks):
            start = i * block_len
            end = start + block_len
            l_res = load_true[start:end] - load_pred_median[start:end]
            p_res = pv_true[start:end] - pv_pred_median[start:end]
            load_res_blocks.append(l_res)
            pv_res_blocks.append(p_res)

        self.load_residuals = np.array(load_res_blocks)
        self.pv_residuals = np.array(pv_res_blocks)
        logger.info(f"Fitted residual pool with {len(self.load_residuals)} blocks of length {block_len}.")
        return self

    def generate_scenarios(
        self,
        load_median: np.ndarray,
        pv_median: np.ndarray,
        solar_zenith: Optional[np.ndarray] = None,
        scenario_count: Optional[int] = None,
        inflation_factor: Optional[float] = None,
        calibration_mode: Optional[str] = None,
        step_seed: Optional[int] = None,
    ) -> ScenarioSet:
        """
        Generate N physically valid 24-hour scenarios around the median point forecasts.
        """
        n_scen = scenario_count if scenario_count is not None else self.unc_cfg.scenario_count
        inflation = inflation_factor if inflation_factor is not None else self.unc_cfg.uncertainty_inflation_factor
        cal_mode = calibration_mode if calibration_mode is not None else self.unc_cfg.calibration_mode
        horizon = len(load_median)

        # Handle calibration dispersion modifier (Section 22)
        if cal_mode == "under_dispersed":
            dispersion_scale = 0.40 * inflation
        elif cal_mode == "over_dispersed":
            dispersion_scale = 1.80 * inflation
        else:  # well_calibrated
            dispersion_scale = 1.00 * inflation

        # Local PRNG for step reproducibility
        rng = np.random.RandomState(step_seed) if step_seed is not None else self.rng

        load_scenarios_kw = []
        load_scenarios_pu = []
        pv_scenarios_pu = []

        # Peak load for pu scaling
        peak_load = 3490.0

        n_pool = len(self.load_residuals)

        for s in range(n_scen):
            if n_pool > 0:
                # Sample a contiguous 24h residual block
                idx = rng.randint(0, n_pool)
                l_res = self.load_residuals[idx][:horizon] * dispersion_scale
                p_res = self.pv_residuals[idx][:horizon] * dispersion_scale
            else:
                # Fallback Gaussian parametric noise if pool is empty
                l_res = rng.normal(0, 0.08 * load_median, size=horizon) * dispersion_scale
                p_res = rng.normal(0, 0.10 * np.maximum(pv_median, 0.05), size=horizon) * dispersion_scale

            # Construct scenarios
            l_scen = load_median + l_res
            p_scen = pv_median + p_res

            # Enforce physical bounds
            l_scen = np.clip(l_scen, 0.0, None)
            p_scen = np.clip(p_scen, 0.0, 1.0)

            # Nighttime PV force zero
            if solar_zenith is not None:
                night = solar_zenith >= 90.0
                p_scen[night] = 0.0

            load_scenarios_kw.append([float(x) for x in l_scen])
            load_scenarios_pu.append([float(x / peak_load) for x in l_scen])
            pv_scenarios_pu.append([float(x) for x in p_scen])

        return ScenarioSet(
            horizon_hours=horizon,
            n_scenarios=n_scen,
            timestamps=[],
            load_scenarios_kw=load_scenarios_kw,
            load_scenarios_pu=load_scenarios_pu,
            pv_scenarios_pu=pv_scenarios_pu,
            load_median_kw=[float(x) for x in load_median],
            load_median_pu=[float(x / peak_load) for x in load_median],
            pv_median_pu=[float(x) for x in pv_median],
            calibration_mode=cal_mode,
            inflation_factor=float(inflation),
            seed=int(step_seed or self.unc_cfg.random_seed),
        )
