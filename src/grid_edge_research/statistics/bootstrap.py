"""
Rigorous statistical inference module for paired controller comparisons.
Computes daily paired differences, block bootstrap 95% confidence intervals,
and Wilcoxon signed-rank significance tests.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

logger = logging.getLogger(__name__)


class PairedComparisonResult(BaseModel):
    """Container for paired statistical test and bootstrap confidence interval results."""
    metric_name: str
    baseline_controller: str
    proposed_controller: str
    n_days: int
    mean_difference: float          # proposed - baseline
    median_difference: float
    ci_95_lower: float
    ci_95_upper: float
    relative_reduction_pct: float   # (baseline - proposed) / baseline * 100
    wilcoxon_stat: Optional[float] = None
    wilcoxon_pvalue: Optional[float] = None
    is_statistically_significant: bool = False


def compute_daily_paired_comparison(
    df_baseline_steps: pd.DataFrame,
    df_proposed_steps: pd.DataFrame,
    metric_col: str = "total_violation_exposure",
    baseline_name: str = "deterministic_mpc",
    proposed_name: str = "stochastic_mpc",
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> PairedComparisonResult:
    """
    Compute daily paired differences and block bootstrap 95% confidence intervals.
    Unit of comparison is calendar day (24-hour block) to avoid autocorrelation confounding.
    """
    # Resample each to daily totals
    daily_base = df_baseline_steps[metric_col].resample("D").sum()
    daily_prop = df_proposed_steps[metric_col].resample("D").sum()

    # Align common days
    common_idx = daily_base.index.intersection(daily_prop.index)
    b_vals = daily_base.loc[common_idx].values
    p_vals = daily_prop.loc[common_idx].values
    n_days = len(b_vals)

    if n_days == 0:
        raise ValueError("No overlapping days found for paired comparison.")

    diffs = p_vals - b_vals  # negative diff means proposed is lower (better for violations)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    base_tot = float(np.sum(b_vals))
    prop_tot = float(np.sum(p_vals))
    rel_reduc = float(((base_tot - prop_tot) / base_tot) * 100.0) if base_tot > 0 else 0.0

    # Block Bootstrap resampling over days
    rng = np.random.RandomState(random_seed)
    boot_means = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_days, size=n_days, replace=True)
        boot_means.append(np.mean(diffs[boot_idx]))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank test
    w_stat, p_val = None, None
    sig = False
    try:
        # Wilcoxon requires non-zero differences
        nz_diffs = diffs[diffs != 0]
        if len(nz_diffs) >= 5:
            res = stats.wilcoxon(nz_diffs, alternative="two-sided")
            w_stat = float(res.statistic)
            p_val = float(res.pvalue)
            sig = bool(p_val < 0.05 and (ci_upper < 0 or ci_lower > 0))
    except Exception as e:
        logger.debug(f"Wilcoxon test skipped or encountered zero diffs: {e}")

    return PairedComparisonResult(
        metric_name=metric_col,
        baseline_controller=baseline_name,
        proposed_controller=proposed_name,
        n_days=n_days,
        mean_difference=mean_diff,
        median_difference=median_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        relative_reduction_pct=rel_reduc,
        wilcoxon_stat=w_stat,
        wilcoxon_pvalue=p_val,
        is_statistically_significant=sig,
    )
