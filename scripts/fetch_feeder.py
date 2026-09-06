"""
Script to download or locate the authoritative IEEE 123-node test feeder files,
verify file integrity, and validate compilation via OpenDSSDirect.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
import requests
import opendssdirect as dss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IEEE123_BASE_URL = "https://raw.githubusercontent.com/tshort/OpenDSS/master/Distrib/IEEETestCases/123Bus/"
REQUIRED_FILES = [
    "IEEE123Master.dss",
    "IEEELineCodes.DSS",
    "IEEE123Loads.DSS",
    "IEEE123Regulators.DSS",
    "BusCoords.dat",
    "Run_IEEE123Bus.DSS",
]


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a local file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def fetch_feeder_files(target_dir: Path | str = "feeder/ieee123") -> Path:
    """Download and verify IEEE 123-bus OpenDSS test files."""
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, dict[str, str]] = {}

    for fname in REQUIRED_FILES:
        out_file = target_path / fname
        url = IEEE123_BASE_URL + fname
        if not out_file.exists():
            logger.info(f"Downloading {fname} from {url}...")
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to download {fname} from {url}. Status code: {resp.status_code}. "
                    f"Please place authoritative IEEE 123-node files manually in {target_path}."
                )
            with open(out_file, "wb") as f:
                f.write(resp.content)
        else:
            logger.info(f"File already present: {out_file}")

        checksum = compute_md5(out_file)
        provenance[fname] = {
            "source_url": url,
            "md5": checksum,
            "size_bytes": str(out_file.stat().st_size),
        }

    # Verify OpenDSS compilation using Text.Command
    master_file = target_path / "IEEE123Master.dss"
    logger.info(f"Verifying OpenDSS compilation with {master_file}...")
    dss.Text.Command("Clear")
    dss.Text.Command(f'Compile "{master_file}"')
    dss.Text.Command("Solve")

    converged = dss.Solution.Converged()
    num_buses = dss.Circuit.NumBuses()
    num_nodes = dss.Circuit.NumNodes()
    logger.info(f"OpenDSS compile & solve success: Converged={converged}, Buses={num_buses}, Nodes={num_nodes}")

    if not converged or num_buses < 100:
        raise RuntimeError(
            f"IEEE 123 test feeder failed validation in OpenDSS: Converged={converged}, Buses={num_buses}"
        )

    # Save provenance metadata
    prov_file = target_path / "provenance.json"
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feeder": "IEEE 123-node Test Feeder",
                "authority": "EPRI / IEEE PES Distribution System Analysis Subcommittee",
                "dss_version": getattr(dss, "__version__", "0.9.x"),
                "files": provenance,
                "validation": {
                    "converged": bool(converged),
                    "num_buses": int(num_buses),
                    "num_nodes": int(num_nodes),
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Provenance saved to {prov_file}")
    return target_path


if __name__ == "__main__":
    try:
        fetch_feeder_files()
        logger.info("IEEE 123 feeder acquisition and verification completed successfully.")
    except Exception as e:
        logger.error(f"Error fetching IEEE 123 feeder: {e}")
        sys.exit(1)
