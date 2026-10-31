
"""Run the EV fleet charging optimisation workflow end-to-end."""

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.io as pio

from config import FleetConfig, ChargingConfig
from data_processor import load_data, prepare_data
from analysis import run_flex_sensitivity
from reporting import (
    build_summary_table,
    plot_charging_profile,
    plot_flex_sensitivity,
)


def run_pipeline(
    fleet_config: FleetConfig | None = None,
    charging_config: ChargingConfig | None = None,
    flex_levels: Sequence[float] | None = None,
    window_index: int = 0,
    method: str = "highs",
) -> dict:
    """
    Load data, optimise charging, analyse results and prepare reports.

    Args:
        fleet_config: Operator-managed fleet specifications and limits.
        charging_config: Customer charging hours and flexibility preferences.
        flex_levels: Sensitivity scenarios as fractions (0.20 = 20%).
            Defaults to 1-percentage-point increments.
        window_index: Charging window to display in the profile chart.
        method: SciPy linear-programming solver.

    Returns:
        Dictionary containing prepared data, selected-scenario results,
        financial and forecast-risk metrics, sensitivity results,
        and Plotly figures.
    """

    fleet_config = fleet_config if fleet_config is not None else FleetConfig()

    charging_config = (
        charging_config if charging_config is not None else ChargingConfig()
    )

    if not 0 <= charging_config.flex_level <= fleet_config.max_flex_level:
        raise ValueError("Customer flexibility exceeds the fleet limit.")

    # Load and prepare settlement-period data
    prices, fleet_demand = load_data()

    core_dataset = prepare_data(
        prices,
        fleet_demand,
        fleet_config.max_fleet_capacity_mwh,
    )

    # Evaluate flexibility in 1-percentage-point increments
    if flex_levels is None:
        flex_levels = np.round(
            np.arange(
                0.01,
                fleet_config.max_flex_level + 1e-9,
                0.01,
            ),
            2,
        ).tolist()

    # Always include the customer's configured flexibility
    scenarios = [*flex_levels, charging_config.flex_level]

    # Run optimisation and financial analysis for each scenario
    flex_results, scenario_outputs = run_flex_sensitivity(
        core_dataset=core_dataset,
        fleet_config=fleet_config,
        charging_config=charging_config,
        flex_levels=scenarios,
        method=method,
    )

    # Retrieve the results for the customer's selected configuration
    selected = scenario_outputs[float(charging_config.flex_level)]

    optimised_dataset = selected["dataset"]
    window_results = selected["window_results"]

    risk_analysis = {
        key: selected[key]
        for key in (
            "risk_metrics",
            "error_symmetry",
            "error_by_sp",
            "error_drivers",
        )
    }

    # Prepare summary table and interactive visualisations
    summary_table = build_summary_table(flex_results)

    figures = {
        "flex_sensitivity": plot_flex_sensitivity(flex_results),
        "charging_profile": plot_charging_profile(
            optimised_dataset=optimised_dataset,
            window_results=window_results,
            window_index=window_index,
        ),
    }

    return {
        "core_dataset": core_dataset,
        "optimised_dataset": optimised_dataset,
        "window_results": window_results,
        "financial_metrics": selected["financial_metrics"],
        "risk_analysis": risk_analysis,
        "flex_results": flex_results,
        "summary_table": summary_table,
        "scenario_outputs": scenario_outputs,
        "figures": figures,
    }


if __name__ == "__main__":

    # Display interactive Plotly charts in the browser
    pio.renderers.default = "browser"

    # Initialise fleet and customer configurations
    fleet_config = FleetConfig()
    charging_config = ChargingConfig()

    # Execute the full optimisation and analysis pipeline
    outputs = run_pipeline(
        fleet_config=fleet_config,
        charging_config=charging_config,
    )

    # Report baseline forecasting risk
    print("\nFORECAST RISK")
    print(
        pd.Series(
            outputs["risk_analysis"]["risk_metrics"]
        ).to_string()
    )

    # Report financial performance for configured flexibility
    print(
        f"\nFINANCIAL PERFORMANCE | "
        f"{charging_config.flex_level:.0%} FLEXIBILITY"
    )

    print(
        pd.Series(
            outputs["financial_metrics"]
        ).to_string()
    )

    print(
        f"\nCHARGING WINDOWS OPTIMISED: "
        f"{len(outputs['window_results']):,}"
    )

    # Report sensitivity across flexibility scenarios
    print("\nFLEXIBILITY SENSITIVITY")
    print(outputs["summary_table"].to_string(index=False))

    # Display interactive charts
    outputs["figures"]["flex_sensitivity"].show()
    outputs["figures"]["charging_profile"].show()
