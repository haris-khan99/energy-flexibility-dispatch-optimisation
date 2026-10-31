"""End-to-end orchestration using synthetic data instead of CSV files."""

import pandas as pd
import pytest
import plotly.graph_objects as go

import src.pipeline


def test_pipeline_returns_consistent_outputs_without_real_csvs(
    core_dataset, fleet_config, charging_config, monkeypatch,
):
    prices = object()
    demand = object()
    calls = []

    monkeypatch.setattr(src.pipeline, "load_data", lambda: (prices, demand))

    def fake_prepare_data(got_prices, got_demand, capacity):
        assert got_prices is prices
        assert got_demand is demand
        calls.append(capacity)
        return core_dataset.copy()

    monkeypatch.setattr(src.pipeline, "prepare_data", fake_prepare_data)

    outputs = src.pipeline.run_pipeline(
        fleet_config=fleet_config,
        charging_config=charging_config,
        flex_levels=[0.0, 0.1],  # 0.2 should be added automatically.
    )

    assert calls == [fleet_config.max_fleet_capacity_mwh]
    assert set(outputs["scenario_outputs"]) == {0.0, 0.1, charging_config.flex_level}
    assert len(outputs["window_results"]) == 2
    assert isinstance(outputs["optimised_dataset"], pd.DataFrame)
    assert outputs["financial_metrics"]["gross_flex_value"] >= 0
    assert outputs["financial_metrics"]["optimised_pnl_benefit"] == pytest.approx(
        outputs["financial_metrics"]["gross_flex_value"], abs=1e-7
    )
    assert isinstance(outputs["figures"]["flex_sensitivity"], go.Figure)
    assert isinstance(outputs["figures"]["charging_profile"], go.Figure)
    assert len(outputs["summary_table"]) == 3


def test_pipeline_rejects_customer_flex_above_fleet_limit(
    fleet_config, charging_config,
):
    from dataclasses import replace

    invalid = replace(
        charging_config, flex_level=fleet_config.max_flex_level + 0.01
    )
    with pytest.raises(ValueError, match="fleet limit"):
        src.pipeline.run_pipeline(
            fleet_config=fleet_config,
            charging_config=invalid,
            flex_levels=[0.1],
        )
