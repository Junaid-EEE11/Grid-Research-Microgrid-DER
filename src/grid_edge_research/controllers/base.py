"""
Base class and action schemas for distribution grid DER controllers.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from grid_edge_research.config import ExperimentConfig
from grid_edge_research.data.feeder_mapping import DERPlacement
from grid_edge_research.powerflow.network_state import NetworkState
from grid_edge_research.uncertainty.scenario_generator import ScenarioSet


class DERControlAction(BaseModel):
    """Control setpoints to apply to physical OpenDSS network for the current time step."""
    pv_curtailment_kw: Dict[str, float] = Field(default_factory=dict)
    pv_q_kvar: Dict[str, float] = Field(default_factory=dict)
    battery_p_kw: Dict[str, float] = Field(default_factory=dict)  # > 0 discharge, < 0 charge
    battery_q_kvar: Dict[str, float] = Field(default_factory=dict)
    solver_status: str = "optimal"
    solve_time_seconds: float = 0.0
    fallback_used: bool = False
    controller_name: str = "base"


class BaseController:
    """Abstract interface for all DER controllers (Rule-based, Deterministic MPC, Risk-Aware MPC)."""

    def __init__(
        self,
        config: ExperimentConfig,
        placement: DERPlacement,
        controller_name: str = "base",
    ):
        self.config = config
        self.placement = placement
        self.controller_name = controller_name

    def compute_action(
        self,
        current_state: NetworkState,
        forecast_load_kw: np.ndarray,
        forecast_pv_pu: np.ndarray,
        scenarios: Optional[ScenarioSet] = None,
        timestamp: Optional[str] = None,
    ) -> DERControlAction:
        """
        Compute optimal or heuristic control setpoint for the next step.
        """
        raise NotImplementedError
