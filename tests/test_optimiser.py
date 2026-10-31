"""Tests of linear-programme feasibility, allocation and charging windows."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.optimiser import create_charging_windows, optimise_energy, run_optimiser


def test_known_optimal_allocation(charging_config):
    window = pd.DataFrame({
        "datetime": pd.date_range("2026-01-05 18:00", periods=4, freq="30min"),
        "forecast_demand_mwh": [10.0] * 4,
        "day_ahead_price": [100.0, 20.0, 80.0, 10.0],
        "physical_capacity_mwh": [20.0] * 4,
    })
    result = optimise_energy(window, replace(charging_config, flex_level=0.20))

    np.testing.assert_allclose(result["optimised_allocation"], [8, 12, 8, 12])
    assert result["baseline_cost"] == pytest.approx(2100.0)
    assert result["optimised_cost"] == pytest.approx(1800.0)
    assert result["gross_flex_value"] == pytest.approx(300.0)
    assert result["shifted"] == pytest.approx(4.0)


def test_energy_conservation_and_all_constraints(core_dataset, charging_config, fleet_config):
    df, results = run_optimiser(core_dataset, charging_config, fleet_config)
    assert len(results) == 2

    for window in results:
        baseline = window["baseline"]
        allocation = window["optimised_allocation"]
        flex = charging_config.flex_level
        assert allocation.sum() == pytest.approx(baseline.sum(), abs=1e-7)
        assert np.all(allocation >= baseline * (1 - flex) - 1e-7)
        assert np.all(allocation <= baseline * (1 + flex) + 1e-7)
        assert window["gross_flex_value"] >= -1e-7

    assert (df["optimised_demand_mwh"] <= df["physical_capacity_mwh"] + 1e-7).all()
    np.testing.assert_allclose(
        df["planned_deviation_mwh"],
        df["optimised_demand_mwh"] - df["forecast_demand_mwh"],
        atol=1e-7,
    )


def test_overnight_grouping_and_daytime_preservation(
    core_dataset, charging_config, fleet_config,
):
    windows = create_charging_windows(core_dataset, charging_config)
    assert len(windows) == 2
    assert all(len(window) == 24 for window in windows)
    assert windows[0]["datetime"].iloc[0] == pd.Timestamp("2026-01-05 18:00")
    assert windows[0]["datetime"].iloc[-1] == pd.Timestamp("2026-01-06 05:30")

    df, _ = run_optimiser(core_dataset, charging_config, fleet_config)
    daytime = (df["datetime"].dt.hour >= 6) & (df["datetime"].dt.hour < 18)
    assert daytime.any()
    np.testing.assert_allclose(
        df.loc[daytime, "optimised_demand_mwh"],
        df.loc[daytime, "forecast_demand_mwh"],
    )
    assert (df.loc[daytime, "planned_deviation_mwh"] == 0).all()


def test_shuffled_rows_are_assigned_to_correct_timestamps(
    core_dataset, charging_config, fleet_config,
):
    ordered, _ = run_optimiser(core_dataset, charging_config, fleet_config)
    shuffled, _ = run_optimiser(
        core_dataset.sample(frac=1, random_state=19),
        charging_config, fleet_config,
    )

    ordered = ordered.sort_values("datetime").reset_index(drop=True)
    shuffled = shuffled.sort_values("datetime").reset_index(drop=True)
    np.testing.assert_allclose(
        ordered["optimised_demand_mwh"], shuffled["optimised_demand_mwh"]
    )
    np.testing.assert_allclose(
        ordered["planned_deviation_mwh"], shuffled["planned_deviation_mwh"]
    )


def test_zero_flexibility_preserves_baseline(core_dataset, charging_config, fleet_config):
    config = replace(charging_config, flex_level=0.0)
    df, results = run_optimiser(core_dataset, config, fleet_config)
    np.testing.assert_allclose(df["optimised_demand_mwh"], df["forecast_demand_mwh"])
    np.testing.assert_allclose(df["planned_deviation_mwh"], 0.0)
    assert sum(r["gross_flex_value"] for r in results) == pytest.approx(0.0, abs=1e-7)


def test_excess_customer_flexibility_is_rejected(
    core_dataset, charging_config, fleet_config,
):
    config = replace(
        charging_config, flex_level=fleet_config.max_flex_level + 0.01
    )
    with pytest.raises(ValueError, match="operator|limit"):
        run_optimiser(core_dataset, config, fleet_config)


def test_infeasible_charging_capacity_raises_error(core_dataset, charging_config):
    window = core_dataset.iloc[:24].copy()
    window["physical_capacity_mwh"] = 5.0  # Below mandatory minimum of 8 MWh.
    with pytest.raises(ValueError, match="Infeasible"):
        optimise_energy(window, charging_config)


def test_incomplete_internal_window_is_rejected(
    core_dataset, charging_config,
):
    # Three sessions: the missing period belongs to the *middle* session.
    df = pd.DataFrame({
        "datetime": pd.date_range(
            "2026-01-05 18:00", "2026-01-08 06:00",
            freq="30min", inclusive="left",
        )
    })
    df = df.loc[df["datetime"] != pd.Timestamp("2026-01-06 20:00")]
    with pytest.raises(ValueError, match="Incomplete|duplicated"):
        create_charging_windows(df, charging_config)


@pytest.mark.parametrize(
    "start,end,expected_periods",
    [
        ("2026-03-28 18:00", "2026-03-29 06:00", 22),
        ("2026-10-24 18:00", "2026-10-25 06:00", 26),
    ],
)
def test_uk_daylight_saving_windows(
    charging_config, start, end, expected_periods,
):
    timestamps = pd.date_range(start, end, freq="30min", inclusive="left", tz="Europe/London")
    data = pd.DataFrame({"datetime": timestamps})
    windows = create_charging_windows(data, charging_config)
    assert len(windows) == 1
    assert len(windows[0]) == expected_periods


def test_physical_capacity_bounds_are_respected(charging_config):
    window = pd.DataFrame({
        "datetime": pd.date_range("2026-01-05 18:00", periods=4, freq="30min"),
        "forecast_demand_mwh": [10.0] * 4,
        "day_ahead_price": [10.0, 20.0, 80.0, 100.0],
        "physical_capacity_mwh": [10.0, 11.0, 11.0, 20.0],
    })
    result = optimise_energy(window, charging_config)
    allocation = result["optimised_allocation"]
    assert allocation.sum() == pytest.approx(40.0)
    assert np.all(allocation <= window["physical_capacity_mwh"] + 1e-7)
    assert allocation[0] == pytest.approx(10.0)
    assert allocation[1] == pytest.approx(11.0)


def test_missing_optimisation_inputs_are_rejected(core_dataset, charging_config):
    window = core_dataset.iloc[:24].copy()
    window.loc[window.index[0], "day_ahead_price"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        optimise_energy(window, charging_config)
