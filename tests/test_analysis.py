"""Tests of forecast-risk, settlement accounting and flexibility sensitivity."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    analyse_forecast_risk,
    calculate_financial_performance,
    run_flex_sensitivity,
)


def test_known_forecast_error_metrics():
    df = pd.DataFrame({
        "forecast_demand_mwh": [10.0, 10.0, 10.0, 10.0],
        "actual_demand_mwh": [12.0, 9.0, 10.0, 7.0],
        "system_price": [50.0, 60.0, 70.0, 80.0],
        "settlement_period": [1, 2, 3, 4],
    })
    results = analyse_forecast_risk(df)
    metrics = results["risk_metrics"]
    assert metrics["observations"] == 4
    assert metrics["mae_mwh"] == pytest.approx(1.5)
    assert metrics["rmse_mwh"] == pytest.approx(np.sqrt(3.5))
    assert metrics["mean_error_mwh"] == pytest.approx(-0.5)
    assert metrics["total_error_settlement_cost"] == pytest.approx(-200.0)
    assert metrics["total_abs_settlement_exposure"] == pytest.approx(400.0)
    assert results["error_symmetry"].loc["underforecast", "observations"] == 1
    assert results["error_symmetry"].loc["overforecast", "observations"] == 2


def test_financial_accounting_matches_hand_calculation():
    df = pd.DataFrame({
        "forecast_demand_mwh": [10.0, 10.0],
        "actual_demand_mwh": [11.0, 9.0],
        "optimised_demand_mwh": [8.0, 12.0],
        "planned_deviation_mwh": [-2.0, 2.0],
        "day_ahead_price": [100.0, 20.0],
        "system_price": [50.0, 60.0],
    })
    out, metrics = calculate_financial_performance(df)
    assert metrics["baseline_total_cost"] == pytest.approx(1190.0)
    assert metrics["optimised_total_cost"] == pytest.approx(1030.0)
    assert metrics["gross_flex_value"] == pytest.approx(160.0)
    assert metrics["optimised_pnl_benefit"] == pytest.approx(160.0)
    assert metrics["shifted_mwh"] == pytest.approx(2.0)
    np.testing.assert_allclose(out["optimised_error_mwh"], out["baseline_error_mwh"])


def test_flex_sensitivity_is_sorted_non_decreasing_and_preserves_preferences(
    core_dataset, fleet_config, charging_config,
):
    original_flex = charging_config.flex_level
    results, scenarios = run_flex_sensitivity(
        core_dataset, fleet_config, charging_config,
        flex_levels=(0.20, 0.0, 0.10, 0.20),
    )
    np.testing.assert_allclose(results["flex_level"], [0.0, 0.1, 0.2])
    assert list(scenarios) == [0.0, 0.1, 0.2]
    assert (results["gross_flex_value"].diff().dropna() >= -1e-7).all()
    assert charging_config.flex_level == original_flex
    assert results.iloc[0]["gross_flex_value"] == pytest.approx(0.0, abs=1e-7)
    assert results.iloc[0]["optimised_pnl_benefit"] == pytest.approx(0.0, abs=1e-7)


def test_sensitivity_rejects_levels_above_operator_limit(
    core_dataset, fleet_config, charging_config,
):
    with pytest.raises(ValueError, match="operator|limit"):
        run_flex_sensitivity(
            core_dataset, fleet_config, charging_config,
            flex_levels=(fleet_config.max_flex_level + 0.05,),
        )
