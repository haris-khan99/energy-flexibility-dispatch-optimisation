"""Reporting tests: structure, values and chart data; no browser required."""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.optimiser import run_optimiser
from src.reporting import (
    build_summary_table,
    plot_charging_profile,
    plot_flex_sensitivity,
)


@pytest.fixture
def flex_results():
    return pd.DataFrame({
        "flex_level": [0.2, 0.1],
        "baseline_total_cost": [2000.0, 2000.0],
        "optimised_total_cost": [1800.0, 1900.0],
        "gross_flex_value": [200.0, 100.0],
        "optimised_pnl_benefit": [200.0, 100.0],
        "marginal_flex_value": [100.0, np.nan],
        "shifted_mwh": [4.0, 2.0],
    })


def test_summary_table_is_sorted_and_converts_fractions_to_percentages(flex_results):
    summary = build_summary_table(flex_results)
    assert summary["Flexibility (%)"].tolist() == [10.0, 20.0]
    assert summary["Net benefit (£)"].tolist() == [100.0, 200.0]
    assert summary["Energy shifted (MWh)"].tolist() == [2.0, 4.0]


def test_sensitivity_plot_contains_expected_values(flex_results):
    fig = plot_flex_sensitivity(flex_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    np.testing.assert_allclose(fig.data[0].x, [10.0, 20.0])
    np.testing.assert_allclose(fig.data[0].y, [100.0, 200.0])


def test_charging_profile_plots_only_selected_window(
    core_dataset, fleet_config, charging_config,
):
    df, windows = run_optimiser(core_dataset, charging_config, fleet_config)
    fig = plot_charging_profile(df, windows, window_index=0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    # The optimiser's window_end is EXCLUSIVE (06:00 is not in 18:00-06:00).
    assert len(fig.data[0].x) == len(windows[0]["baseline"])
    assert len(fig.data[1].x) == len(windows[0]["baseline"])


def test_profile_rejects_unknown_window(core_dataset, fleet_config, charging_config):
    df, windows = run_optimiser(core_dataset, charging_config, fleet_config)
    with pytest.raises(IndexError):
        plot_charging_profile(df, windows, window_index=len(windows))
