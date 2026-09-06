"""
Deterministic spatial mapping of PV systems and battery energy storage across
IEEE 123-node feeder buses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel

from grid_edge_research.config import ExperimentConfig, load_config

logger = logging.getLogger(__name__)

# IEEE 123 Nominal reference feeder peak load (kW)
IEEE123_NOMINAL_TOTAL_KW = 3490.0

# Authoritative subset of IEEE 123 load buses with good 3-phase and 1-phase spatial dispersion
DEFAULT_CANDIDATE_PV_BUSES = [
    "1", "2", "4", "7", "9", "11", "13", "18", "23", "28",
    "35", "40", "48", "52", "60", "65", "76", "87", "98", "114"
]


class DERPlacement(BaseModel):
    """Immutable record of DER allocations across feeder buses."""
    pv_penetration_level: float
    total_pv_installed_kw: float
    pv_buses: List[str]
    pv_ratings_kw: Dict[str, float]
    battery_buses: List[str]
    battery_ratings_kw: Dict[str, float]
    battery_ratings_kwh: Dict[str, float]
    seed_used: int


def determine_der_placement(
    config: Optional[ExperimentConfig] = None,
    pv_penetration: Optional[float] = None,
    save_path: Optional[Path | str] = None,
) -> DERPlacement:
    """
    Select PV buses and battery locations deterministically based on configuration.
    
    PV Penetration is defined as:
        Penetration = Total Installed PV Nameplate (kW) / Nominal Reference Feeder Load (3490 kW)
    """
    if config is None:
        config = load_config()

    pen = pv_penetration if pv_penetration is not None else config.pv.penetration_level
    seed = config.pv.pv_bus_seed
    candidate_buses = DEFAULT_CANDIDATE_PV_BUSES

    total_pv_kw = pen * IEEE123_NOMINAL_TOTAL_KW

    # Select PV buses reproducibly using configured seed
    rng = np.random.RandomState(seed)
    
    if pen <= 0.0:
        pv_buses = []
        pv_ratings = {}
    else:
        # Choose a subset of 20 distributed load buses for PV installation
        n_pv_buses = min(config.pv.candidate_buses_count, len(candidate_buses))
        chosen_indices = rng.choice(len(candidate_buses), size=n_pv_buses, replace=False)
        chosen_indices.sort()
        pv_buses = [str(candidate_buses[i]) for i in chosen_indices]

        # Allocate PV capacity across buses with modest non-uniformity (realistic rooftop variations)
        proportions = rng.dirichlet(np.ones(n_pv_buses) * 4)
        pv_ratings = {bus: float(round(p * total_pv_kw, 2)) for bus, p in zip(pv_buses, proportions)}

    # Battery placement
    batt_cfg = config.battery
    battery_buses = [str(b) for b in batt_cfg.bus_placements]
    battery_ratings_kw = {b: batt_cfg.p_rating_kw_per_batt for b in battery_buses}
    battery_ratings_kwh = {b: batt_cfg.e_rating_kwh_per_batt for b in battery_buses}

    placement = DERPlacement(
        pv_penetration_level=float(pen),
        total_pv_installed_kw=float(round(total_pv_kw, 2)),
        pv_buses=pv_buses,
        pv_ratings_kw=pv_ratings,
        battery_buses=battery_buses,
        battery_ratings_kw=battery_ratings_kw,
        battery_ratings_kwh=battery_ratings_kwh,
        seed_used=seed,
    )

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(placement.model_dump_json(indent=2))
        logger.info(f"Saved DER placement artifact to {out_p}")

    return placement
