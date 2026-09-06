"""
Script to estimate and validate OpenDSS voltage sensitivity matrices (S_P, S_Q)
and store the resulting linear network approximation model.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_edge_research.config import load_config
from grid_edge_research.data.feeder_mapping import determine_der_placement
from grid_edge_research.optimization.sensitivity_model import estimate_feeder_sensitivities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str | None = None, output_path: str = "results/models/sensitivity_model.json") -> None:
    cfg = load_config(config_path)
    placement = determine_der_placement(cfg, save_path="results/models/der_placement.json")
    
    out_file = Path(output_path).resolve()
    logger.info("Starting OpenDSS voltage sensitivity estimation...")
    model = estimate_feeder_sensitivities(
        config=cfg,
        placement=placement,
        perturbation_p_kw=cfg.sensitivity.perturbation_p_kw,
        perturbation_q_kvar=cfg.sensitivity.perturbation_q_kvar,
        save_path=out_file,
    )
    print("\n" + "=" * 60)
    print("VOLTAGE SENSITIVITY ESTIMATION SUMMARY")
    print("=" * 60)
    print(f"Observed Network Nodes: {len(model.node_names)}")
    print(f"PV Controllable Buses: {len(model.pv_buses)}")
    print(f"Battery Controllable Buses: {len(model.battery_buses)}")
    print(f"Validation MAE Error: {model.mae_validation_error:.6f} pu")
    print(f"Validation Max Error: {model.max_validation_error:.6f} pu")
    print(f"Saved artifact to: {out_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
