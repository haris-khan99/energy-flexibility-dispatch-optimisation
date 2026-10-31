
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from config import FleetConfig, ChargingConfig
from data_processor import load_data, prepare_data
from analysis import run_flex_sensitivity

pio.renderers.default = "browser" 

def build_summary_table(flex_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build a formatted summary of flexibility scenario performance.

    Args:
        flex_results: Scenario results from run_flex_sensitivity().

    Returns:
        DataFrame containing financial and flexibility metrics,
        with readable column names and rounded numeric values.
    """

    columns = {
        "flex_level": "Flexibility (%)",
        "baseline_total_cost": "Baseline cost (£)",
        "optimised_total_cost": "Optimised cost (£)",
        "gross_flex_value": "Gross flex value (£)",
        "optimised_pnl_benefit": "Net benefit (£)",
        "marginal_flex_value": "Marginal flex value (£)",
        "shifted_mwh": "Energy shifted (MWh)",
    }

    missing = set(columns) - set(flex_results.columns)

    if missing:
        raise ValueError(
            f"Flexibility results missing columns: {sorted(missing)}"
        )

    summary = (
        flex_results.loc[:, list(columns)]
        .sort_values("flex_level")
        .rename(columns=columns)
        .reset_index(drop=True)
    )

    summary["Flexibility (%)"] = (
        summary["Flexibility (%)"] * 100
    ).round(1)

    numeric_columns = summary.columns.drop("Flexibility (%)")
    summary[numeric_columns] = summary[numeric_columns].round(2)

    return summary


def plot_charging_profile(
    optimised_dataset: pd.DataFrame,
    window_results: list[dict],
    window_index: int = 0,
) -> go.Figure:
    """
    Visualise baseline and optimised EV charging demand alongside
    day-ahead electricity prices for a selected charging window.

    Args:
        optimised_dataset: Settlement-period optimisation output.
        window_results: Individual charging-window optimisation results.
        window_index: Index of the charging window to visualise.

    Returns:
        Interactive Plotly figure comparing charging demand and prices.
    """

    if not 0 <= window_index < len(window_results):
        raise IndexError(
            "window_index is outside the available charging windows."
        )

    window = window_results[window_index]

    start = pd.Timestamp(window["window_start"])
    end = pd.Timestamp(window["window_end"])

    required = {
        "datetime",
        "forecast_demand_mwh",
        "optimised_demand_mwh",
        "day_ahead_price",
    }

    missing = required - set(optimised_dataset.columns)

    if missing:
        raise ValueError(
            f"Optimised dataset missing columns: {sorted(missing)}"
        )

    df = optimised_dataset.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.loc[
        df["datetime"].between(start, end, inclusive="both")
    ].sort_values("datetime")

    if df.empty or df["optimised_demand_mwh"].isna().any():
        raise ValueError(
            "Selected charging window has no complete optimised allocations."
        )

    # Create dual-axis figure for charging demand and electricity prices
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # Baseline charging demand
    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["forecast_demand_mwh"],
            name="Baseline demand",
            hovertemplate=(
                "%{x}<br>Baseline: %{y:.2f} MWh<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Optimised charging demand
    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["optimised_demand_mwh"],
            name="Optimised demand",
            hovertemplate=(
                "%{x}<br>Optimised: %{y:.2f} MWh<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Day-ahead electricity prices
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["day_ahead_price"],
            name="Day-ahead price",
            mode="lines+markers",
            line={"width": 2},
            hovertemplate=(
                "%{x}<br>DA price: £%{y:.2f}/MWh<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=(
            f"Charging profile | Window starting "
            f"{start:%d %b %Y, %H:%M}"
        ),
        xaxis_title="Settlement period",
        barmode="group",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"t": 110, "r": 60, "b": 60, "l": 65},
        template="plotly_white",
    )

    fig.update_yaxes(
        title_text="Charging demand (MWh)",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Day-ahead price (£/MWh)",
        secondary_y=True,
    )

    return fig


def plot_flex_sensitivity(
    flex_results: pd.DataFrame,
) -> go.Figure:
    """
    Visualise the financial benefit of different flexibility scenarios.

    Args:
        flex_results: Results returned by run_flex_sensitivity().

    Returns:
        Interactive Plotly figure showing financial benefit
        against permitted charging flexibility.
    """

    required = {
        "flex_level",
        "optimised_pnl_benefit",
        "gross_flex_value",
        "shifted_mwh",
    }

    missing = required - set(flex_results.columns)

    if missing:
        raise ValueError(
            f"Flexibility results missing columns: {sorted(missing)}"
        )

    df = flex_results.sort_values("flex_level")

    flex_percentage = df["flex_level"] * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=flex_percentage,
            y=df["optimised_pnl_benefit"],
            mode="lines+markers",
            name="Net financial benefit",
            customdata=df[
                ["gross_flex_value", "shifted_mwh"]
            ].to_numpy(),
            hovertemplate=(
                "Flexibility: %{x:.0f}%<br>"
                "Net benefit: £%{y:,.2f}<br>"
                "Gross flex value: £%{customdata[0]:,.2f}<br>"
                "Energy shifted: %{customdata[1]:,.2f} MWh"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Financial benefit by charging flexibility",
        xaxis_title="Permitted flexibility (%)",
        yaxis_title="Net financial benefit (£)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


if __name__ == "__main__":

    # Initialise configurations
    fleet_config = FleetConfig()
    charging_config = ChargingConfig()

    # Load and prepare data
    prices, fleet_demand = load_data()

    core_dataset = prepare_data(
        prices,
        fleet_demand,
        fleet_config.max_fleet_capacity_mwh,
    )

    flex_levels = np.round(np.arange(0.01, 0.401, 0.01), 2)

    # Run financial sensitivity analysis
    flex_results, scenario_outputs = run_flex_sensitivity(
        core_dataset=core_dataset,
        fleet_config=fleet_config,
        charging_config=charging_config,
        flex_levels=flex_levels
    )

    # Present financial summary
    summary_table = build_summary_table(flex_results)

    print("\nFLEXIBILITY SUMMARY")
    print(summary_table.to_string(index=False))

    # Display flexibility sensitivity chart
    plot_flex_sensitivity(flex_results).show()

    # Display a charging profile for the customer's selected flexibility
    chosen_flex = charging_config.flex_level

    if chosen_flex in scenario_outputs:

        scenario = scenario_outputs[chosen_flex]

        if scenario["window_results"]:
            plot_charging_profile(
                optimised_dataset=scenario["dataset"],
                window_results=scenario["window_results"],
                window_index=0,
            ).show()
