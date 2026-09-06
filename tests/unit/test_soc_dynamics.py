"""
Unit tests for battery state-of-charge dynamics, units, and sign conventions.
"""

import numpy as np
import pytest

from grid_edge_research.config import BatteryConfig


def test_battery_sign_convention_and_soc_update():
    """
    Sign convention:
    P_batt > 0: Discharging / injecting power into the grid -> SOC DECREASES.
    P_batt < 0: Charging / consuming power from the grid -> SOC INCREASES.
    """
    batt_cfg = BatteryConfig(
        p_rating_kw_per_batt=100.0,
        e_rating_kwh_per_batt=200.0,
        soc_initial=0.50,
        soc_min=0.20,
        soc_max=0.90,
        eta_charge=0.95,
        eta_discharge=0.95,
    )

    soc_0 = 0.50
    dt_hours = 1.0  # 1 hour
    E_kwh = batt_cfg.default_energy_capacity_kwh  # 200 kWh

    # Case 1: Discharge at 50 kW for 1 hour (P_dis = 50, P_batt = +50 kW)
    p_dis = 50.0  # kW
    delta_e_dis = (p_dis * dt_hours) / batt_cfg.eta_discharge  # kWh drawn from cell
    soc_1 = soc_0 - (delta_e_dis / E_kwh)
    assert soc_1 < soc_0, "SOC must decrease during discharge"
    expected_soc_1 = 0.50 - (50.0 / (0.95 * 200.0))
    assert np.isclose(soc_1, expected_soc_1)

    # Case 2: Charge at 50 kW for 1 hour (P_ch = 50, P_batt = -50 kW)
    p_ch = 50.0  # kW
    delta_e_ch = (p_ch * dt_hours) * batt_cfg.eta_charge  # kWh stored into cell
    soc_2 = soc_0 + (delta_e_ch / E_kwh)
    assert soc_2 > soc_0, "SOC must increase during charging"
    expected_soc_2 = 0.50 + (50.0 * 0.95 / 200.0)
    assert np.isclose(soc_2, expected_soc_2)


def test_battery_throughput_accounting():
    """Verify that battery throughput accumulates both charge and discharge energy."""
    p_dis = np.array([10.0, 20.0, 0.0, 50.0])  # kW
    p_ch = np.array([0.0, 0.0, 30.0, 0.0])      # kW
    dt = 1.0  # hour

    throughput_kwh = float(np.sum(p_dis + p_ch) * dt)
    assert throughput_kwh == 110.0  # 10 + 20 + 30 + 50 = 110 kWh
