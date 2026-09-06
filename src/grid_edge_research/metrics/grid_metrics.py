"""
Comprehensive grid performance and reliability metrics computation.
Computes voltage violation exposure, curtailment, battery throughput/cycles,
losses, peak import, and optimization health.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel


class ExperimentMetrics(BaseModel):
    """Aggregated summary metrics for a closed-loop simulation experiment."""
    experiment_id: str
    controller: str
    pv_penetration_pct: float
    uncertainty_level: str
    total_hours_simulated: int
    
    # Primary Reliability Endpoint
    total_voltage_violation_exposure: float
    bus_node_hours_violation_fraction: float
    max_voltage_pu: float
    min_voltage_pu: float
    p95_voltage_deviation_pu: float
    p99_voltage_deviation_pu: float

    # Energy and DER Metrics
    total_pv_generated_kwh: float
    total_pv_curtailment_kwh: float
    curtailment_percentage: float
    battery_throughput_kwh: float
    battery_equivalent_cycles: float
    total_feeder_losses_kwh: float
    substation_peak_kw: float
    total_net_energy_import_kwh: float

    # Solver Health
    optimization_failures_count: int
    fallbacks_count: int
    opendss_convergence_failures_count: int
    mean_solve_time_sec: float


def compute_simulation_metrics(
    step_records_df: pd.DataFrame,
    experiment_id: str = "run_001",
    controller: str = "stochastic_mpc",
    pv_penetration_pct: float = 50.0,
    uncertainty_level: str = "nominal",
    nominal_battery_capacity_kwh: float = 800.0,
) -> ExperimentMetrics:
    """
    Compute comprehensive physical grid and DER operation metrics from step records.
    """
    n_hours = len(step_records_df)
    if n_hours == 0:
        raise ValueError("Step records DataFrame is empty.")

    # 1. Voltage Violations
    total_viol = float(step_records_df["total_violation_exposure"].sum())
    v_max = float(step_records_df["v_max_pu"].max())
    v_min = float(step_records_df["v_min_pu"].min())
    
    # Node violation fraction: sum(nodes_violating) / sum(total_nodes)
    tot_viol_nodes = float(step_records_df["nodes_violating_count"].sum())
    tot_nodes_sampled = float(step_records_df["total_node_count"].sum())
    node_frac = tot_viol_nodes / tot_nodes_sampled if tot_nodes_sampled > 0 else 0.0

    # Max deviations from 1.0 pu
    dev_max = np.maximum(step_records_df["v_max_pu"] - 1.0, 1.0 - step_records_df["v_min_pu"])
    p95_dev = float(np.percentile(dev_max, 95))
    p99_dev = float(np.percentile(dev_max, 99))

    # 2. PV and Curtailment
    pv_gen_kwh = float(step_records_df["pv_gen_total_kw"].sum())
    pv_curt_kwh = float(step_records_df["pv_curt_total_kw"].sum())
    tot_pv_avail = pv_gen_kwh + pv_curt_kwh
    curt_pct = float((pv_curt_kwh / tot_pv_avail) * 100.0) if tot_pv_avail > 0 else 0.0

    # 3. Battery Throughput and Equivalent Cycles
    batt_abs_kwh = float(step_records_df["battery_abs_power_kw"].sum())
    batt_throughput_kwh = batt_abs_kwh
    batt_cycles = batt_abs_kwh / (2.0 * nominal_battery_capacity_kwh) if nominal_battery_capacity_kwh > 0 else 0.0

    # 4. Losses, Peak Substation Import, Net Energy
    losses_kwh = float(step_records_df["feeder_losses_kw"].sum())
    sub_peak_kw = float(step_records_df["substation_p_kw"].max())
    net_import_kwh = float(step_records_df["substation_p_kw"].sum())

    # 5. Solver Health
    opt_fails = int((step_records_df["solver_status"] != "optimal").sum())
    fallbacks = int(step_records_df["fallback_used"].sum())
    dss_fails = int((~step_records_df["converged"]).sum())
    mean_solve_time = float(step_records_df["solve_time_seconds"].mean())

    return ExperimentMetrics(
        experiment_id=experiment_id,
        controller=controller,
        pv_penetration_pct=pv_penetration_pct,
        uncertainty_level=uncertainty_level,
        total_hours_simulated=n_hours,
        total_voltage_violation_exposure=total_viol,
        bus_node_hours_violation_fraction=node_frac,
        max_voltage_pu=v_max,
        min_voltage_pu=v_min,
        p95_voltage_deviation_pu=p95_dev,
        p99_voltage_deviation_pu=p99_dev,
        total_pv_generated_kwh=pv_gen_kwh,
        total_pv_curtailment_kwh=pv_curt_kwh,
        curtailment_percentage=curt_pct,
        battery_throughput_kwh=batt_throughput_kwh,
        battery_equivalent_cycles=batt_cycles,
        total_feeder_losses_kwh=losses_kwh,
        substation_peak_kw=sub_peak_kw,
        total_net_energy_import_kwh=net_import_kwh,
        optimization_failures_count=opt_fails,
        fallbacks_count=fallbacks,
        opendss_convergence_failures_count=dss_fails,
        mean_solve_time_sec=mean_solve_time,
    )
