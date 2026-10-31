"""Tests for fleet specifications and charging preferences."""

import pytest


def test_fleet_capacity_matches_charger_specifications(fleet_config):
    expected_mwh = (
        fleet_config.n_chargers
        * fleet_config.charger_power_kw
        * fleet_config.settlement_period_hours
        / 1_000
    )
    assert fleet_config.max_fleet_capacity_mwh == pytest.approx(expected_mwh)


def test_default_charging_preferences_respect_operator_limits(
    charging_config, fleet_config,
):
    assert charging_config.start_time != charging_config.end_time
    assert 0 <= charging_config.flex_level <= fleet_config.max_flex_level <= 1
