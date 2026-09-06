"""
Unit tests for configuration validation and parameter integrity.
"""

import pytest
from pydantic import ValidationError

from grid_edge_research.config import (
    BatteryConfig,
    ExperimentConfig,
    PVConfig,
    SmartInverterConfig,
    VoltageLimitsConfig,
    load_config,
)


def test_default_config_loads_valid():
    cfg = load_config()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.voltage_limits.v_min_pu == 0.95
    assert cfg.voltage_limits.v_max_pu == 1.05
    assert cfg.battery.soc_min < cfg.battery.soc_max
    assert cfg.pv.penetration_level in [0.0, 0.25, 0.50, 0.75, 1.0]


def test_invalid_voltage_limits_raises():
    with pytest.raises(ValidationError):
        VoltageLimitsConfig(v_min_pu=1.05, v_max_pu=0.95)


def test_battery_efficiencies_and_bounds():
    with pytest.raises(ValidationError):
        BatteryConfig(eta_charge=1.2, eta_discharge=0.95)

    with pytest.raises(ValidationError):
        BatteryConfig(soc_min=1.5, soc_max=0.9)


def test_inverter_oversize_factor():
    cfg = SmartInverterConfig(s_oversize_factor=1.15)
    assert cfg.s_oversize_factor == 1.15
