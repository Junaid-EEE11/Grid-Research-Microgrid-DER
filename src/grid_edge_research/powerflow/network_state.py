"""
Data classes representing physical distribution network states and snapshot metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field


class BusVoltage(BaseModel):
    """Voltage measurement at a specific bus and phase."""
    bus_id: str
    phase: int
    v_mag_pu: float
    v_mag_kv: float
    v_ang_deg: float


class NetworkState(BaseModel):
    """Full snapshot state of the distribution network from OpenDSS AC power flow."""
    timestamp: Optional[str] = None
    converged: bool
    iterations: int
    
    # Voltages
    bus_voltages_pu: Dict[str, float] = Field(default_factory=dict)  # "bus.node" -> V_pu
    v_min_pu: float
    v_max_pu: float
    v_mean_pu: float
    total_violation_exposure: float  # sum(max(0, V - 1.05) + max(0, 0.95 - V))
    nodes_violating_count: int
    total_node_count: int

    # Power and Losses
    substation_p_kw: float
    substation_q_kvar: float
    feeder_losses_kw: float
    feeder_losses_kvar: float
    
    # DER Injections
    pv_generation_kw: Dict[str, float] = Field(default_factory=dict)
    pv_curtailment_kw: Dict[str, float] = Field(default_factory=dict)
    pv_reactive_kvar: Dict[str, float] = Field(default_factory=dict)
    battery_p_kw: Dict[str, float] = Field(default_factory=dict)  # > 0 discharge, < 0 charge
    battery_soc: Dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
