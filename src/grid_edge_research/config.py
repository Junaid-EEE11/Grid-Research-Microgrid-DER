"""
Validated configuration models and YAML parser using Pydantic.
Enforces rigorous validation across all experiment parameters, physics bounds,
and random seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class DateRangeConfig(BaseModel):
    """Chronological dataset splitting date boundaries."""
    train_start: str = "2012-01-01 00:00:00"
    train_end: str = "2013-09-30 23:00:00"
    val_start: str = "2013-10-01 00:00:00"
    val_end: str = "2013-12-31 23:00:00"
    test_start: str = "2014-01-01 00:00:00"
    test_end: str = "2014-12-31 23:00:00"


class LocationConfig(BaseModel):
    """Geographic coordinates for solar irradiance & weather data (Portugal)."""
    name: str = "Portugal_Representative_Site"
    latitude: float = 39.50
    longitude: float = -8.00
    timezone: str = "UTC"


class BatteryConfig(BaseModel):
    """Battery energy storage system parameters."""
    bus_placements: List[str] = Field(default_factory=lambda: ["13", "48", "65", "114"])
    p_rating_kw_per_batt: float = 100.0
    e_rating_kwh_per_batt: float = 200.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min: float = 0.20
    soc_max: float = 0.90
    soc_initial: float = 0.50

    @property
    def default_power_rating_kw(self) -> float:
        return self.p_rating_kw_per_batt

    @property
    def default_energy_capacity_kwh(self) -> float:
        return self.e_rating_kwh_per_batt

    @field_validator("eta_charge", "eta_discharge")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Efficiency must be in (0, 1], got {v}")
        return v

    @field_validator("soc_min", "soc_max", "soc_initial")
    @classmethod
    def validate_soc_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"SOC must be in [0, 1], got {v}")
        return v


class SmartInverterConfig(BaseModel):
    """PV smart-inverter parameters."""
    s_oversize_factor: float = 1.10  # S_rated = 1.10 * P_rated
    q_max_ratio: float = 0.44        # Maximum reactive power capability ratio
    volt_var_v1: float = 0.95
    volt_var_v2: float = 0.98
    volt_var_v3: float = 1.02
    volt_var_v4: float = 1.05


class PVConfig(BaseModel):
    """Photovoltaic distribution and penetration parameters."""
    penetration_level: float = 0.50  # 0.0, 0.25, 0.50, 0.75, 1.00
    pv_bus_seed: int = 42
    candidate_buses_count: int = 20
    inverter: SmartInverterConfig = Field(default_factory=SmartInverterConfig)

    @field_validator("penetration_level")
    @classmethod
    def validate_penetration(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"PV penetration must be in [0.0, 2.0], got {v}")
        return v


class ForecastingConfig(BaseModel):
    """Forecasting task parameters."""
    horizon_hours: int = 24
    forecast_horizon_hours: int = 24
    control_interval_hours: float = 1.0
    quantiles: List[float] = Field(default_factory=lambda: [0.10, 0.50, 0.90])
    load_model_type: Literal["persistence", "point_gbr", "quantile_gbr"] = "quantile_gbr"
    pv_model_type: Literal["persistence", "point_gbr", "quantile_gbr"] = "quantile_gbr"
    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.05
    min_samples_leaf: int = 10


class UncertaintyConfig(BaseModel):
    """Scenario generation & calibration configuration."""
    scenario_count: int = 50
    method: Literal["block_bootstrap", "residual_gaussian", "quantile_sampling"] = "block_bootstrap"
    block_length_hours: int = 24
    uncertainty_inflation_factor: float = 1.0  # 1.0=nominal, 1.5=moderate, 2.0=strong
    stress_test_mode: str = "nominal"
    calibration_mode: Literal["well_calibrated", "under_dispersed", "over_dispersed"] = "well_calibrated"
    cvar_alpha: float = 0.95
    random_seed: int = 42


class VoltageLimitsConfig(BaseModel):
    """Acceptable per-unit voltage limits."""
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    v_nom_pu: float = 1.00

    @field_validator("v_max_pu")
    @classmethod
    def validate_voltage_limits(cls, v_max: float, info) -> float:
        v_min = info.data.get("v_min_pu", 0.95)
        if v_min >= v_max:
            raise ValueError(f"v_min_pu ({v_min}) must be strictly less than v_max_pu ({v_max})")
        return v_max


class ObjectiveWeightsConfig(BaseModel):
    """Normalized objective function weights for optimization."""
    w_voltage_violation: float = 1000.0   # Penalty for expected voltage violation (pu*h)
    w_cvar_risk: float = 2000.0          # Penalty for tail CVaR voltage violation
    w_voltage_deviation: float = 10.0     # Small penalty for deviation from 1.0 pu
    w_pv_curtailment: float = 50.0        # Cost per kWh of PV curtailed
    w_battery_wear: float = 2.0           # Cost per kWh of battery throughput (degradation proxy)
    w_reactive_power: float = 0.5         # Penalty for inverter reactive power
    w_energy_import: float = 0.15         # Cost per kWh of substation import
    w_energy_export: float = 0.05         # Reward per kWh of substation export


class SensitivityConfig(BaseModel):
    """Voltage sensitivity matrix estimation configuration."""
    perturbation_p_kw: float = 25.0
    perturbation_q_kvar: float = 25.0
    reference_load_scaling: float = 1.0
    validation_sample_count: int = 20


class OptimizationConfig(BaseModel):
    """Optimization solver parameters."""
    horizon_hours: int = 24
    cvar_alpha: float = 0.95
    solver: Literal["CLARABEL", "OSQP", "SCS", "HIGHS", "SCIPY"] = "CLARABEL"
    solver_verbose: bool = False
    weights: ObjectiveWeightsConfig = Field(default_factory=ObjectiveWeightsConfig)
    enable_reactive_power: bool = True
    enable_battery: bool = True


class PathsConfig(BaseModel):
    """File path configurations."""
    feeder_dir: Path = Field(default=Path("feeder/ieee123"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    interim_data_dir: Path = Field(default=Path("data/interim"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    results_dir: Path = Field(default=Path("results"))
    reports_dir: Path = Field(default=Path("reports"))


class ExperimentConfig(BaseModel):
    """Root configuration model containing all sub-configurations."""
    experiment_id: str = "smoke_experiment"
    description: str = "Default baseline research experiment configuration"
    random_seed: int = 42
    controller_type: Literal["rule_based", "deterministic_mpc", "stochastic_mpc"] = "stochastic_mpc"
    
    date_ranges: DateRangeConfig = Field(default_factory=DateRangeConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    pv: PVConfig = Field(default_factory=PVConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    voltage_limits: VoltageLimitsConfig = Field(default_factory=VoltageLimitsConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @property
    def controller(self) -> str:
        return self.controller_type

    @controller.setter
    def controller(self, value: str) -> None:
        self.controller_type = value  # type: ignore

    @property
    def objective_weights(self) -> ObjectiveWeightsConfig:
        return self.optimization.weights

    def to_yaml(self, path: str | Path) -> None:
        """Write resolved config to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[str | Path] = None) -> ExperimentConfig:
    """Load and validate configuration from YAML or return default."""
    if config_path is None:
        default_path = Path("configs/base.yaml")
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return ExperimentConfig(**data)
        return ExperimentConfig()

    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ExperimentConfig(**data)
