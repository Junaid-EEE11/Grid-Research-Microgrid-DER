"""
Feeder topology parsing, phase connection helper, and DER generator injection setup.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import opendssdirect as dss

logger = logging.getLogger(__name__)


def get_bus_phase_spec(bus_id: str) -> Tuple[str, int, float]:
    """
    Query the active OpenDSS circuit to determine the available phases and nominal voltage of a bus.
    
    Returns:
        bus_spec: e.g. "13.1.2.3" or "114.1"
        phases: 1 or 3
        kv: 4.16 for 3-phase wye, 2.4 for 1-phase line-to-neutral
    """
    dss.Circuit.SetActiveBus(str(bus_id))
    nodes = list(dss.Bus.Nodes())
    # Filter out ground (node 0)
    phase_nodes = [n for n in nodes if n in (1, 2, 3)]
    
    if len(phase_nodes) == 3:
        bus_spec = f"{bus_id}.1.2.3"
        phases = 3
        kv = 4.16
    elif len(phase_nodes) == 1:
        bus_spec = f"{bus_id}.{phase_nodes[0]}"
        phases = 1
        kv = 2.4
    elif len(phase_nodes) == 2:
        # 2-phase lateral: connect on primary phase node
        bus_spec = f"{bus_id}.{phase_nodes[0]}"
        phases = 1
        kv = 2.4
    else:
        # Fallback single phase 1
        bus_spec = f"{bus_id}.1"
        phases = 1
        kv = 2.4

    return bus_spec, phases, kv
