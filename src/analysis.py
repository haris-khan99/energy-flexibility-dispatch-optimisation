
import numpy as np
import pandas as pd

from dataclasses import replace
from typing import Sequence

from config import FleetConfig, ChargingConfig
from optimiser import run_optimiser
from data_processor import load_data, prepare_data


def analyse_forecast_risk(dataset: pd.DataFrame) -> dict:
    """
    Analyse baseline EV demand forecast accuracy and settlement exposure.

    Calculates forecast error metrics, directional bias, settlement
    exposure and correlations with potential forecast error drivers.

    Args:
        dataset: Settlement-period dataset containing forecast demand,
            actual demand, system prices and relevant features.

    Returns:
        Dictionary containing forecast risk metrics, directional error
        analysis, settlement-period errors and error-driver correlations.
    """

    df = dataset.copy()

    # Calculate baseline forecast errors
    df["baseline_error_mwh"] = (
        df["actual_demand_mwh"] - df["forecast_demand_mwh"]
    )

    df["abs_baseline_error_mwh"] = df["baseline_error_mwh"].abs()

    # Calculate settlement exposure from forecast deviations
    df["baseline_error_cost"] = (
        df["baseline_error_mwh"] * df["system_price"]
    )

    df["abs_settlement_exposure"] = df["baseline_error_cost"].abs()

    # Calculate headline forecast risk metrics
    error = df["baseline_error_mwh"].to_numpy(dtype=float)

    risk_metrics = {
        "observations": len(df),
        "mae_mwh": float(np.abs(error).mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error ** 2))),
        "mean_error_mwh": float(np.mean(error)),
        "total_error_settlement_cost": float(
            df["baseline_error_cost"].sum()
        ),
        "total_abs_settlement_exposure": float(
            df["abs_settlement_exposure"].sum()
        ),
    }

    # Identify underforecasting and overforecasting periods
    df["error_direction"] = np.select(
        [
            df["baseline_error_mwh"] > 0,
            df["baseline_error_mwh"] < 0,
        ],
        [
            "underforecast",
            "overforecast",
        ],
        default="correct",
    )

    # Analyse forecast errors by direction
    error_symmetry = df.groupby("error_direction").agg(
        observations=("baseline_error_mwh", "count"),
        mean_error_mwh=("baseline_error_mwh", "mean"),
        mean_abs_error_mwh=("abs_baseline_error_mwh", "mean"),
        mean_system_price=("system_price", "mean"),
        total_error_settlement_cost=("baseline_error_cost", "sum"),
        total_abs_settlement_exposure=("abs_settlement_exposure", "sum"),
    )

    # Identify settlement periods with the greatest forecast errors
    error_by_sp = (
        df.groupby("settlement_period")
        .agg(
            mean_error_mwh=("baseline_error_mwh", "mean"),
            mae_mwh=("abs_baseline_error_mwh", "mean"),
        )
        .reset_index()
    )

    # Examine potential drivers of absolute forecast error
    candidate_drivers = [
        "forecast_demand_mwh",
        "plugged_in_fraction",
        "temperature_c",
    ]

    available_drivers = [
        col for col in candidate_drivers if col in df.columns
    ]

    correlation_columns = [
        "abs_baseline_error_mwh",
        *available_drivers,
    ]

    error_drivers = (
        df[correlation_columns]
        .corr(numeric_only=True)["abs_baseline_error_mwh"]
        .drop("abs_baseline_error_mwh")
        .rename("correlation_with_absolute_error")
    )

    return {
        "risk_metrics": risk_metrics,
        "error_symmetry": error_symmetry,
        "error_by_sp": error_by_sp,
        "error_drivers": error_drivers,
    }


def calculate_financial_performance(
    optimised_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Compare baseline and optimised electricity procurement costs.

    Calculates day-ahead procurement costs, imbalance settlement costs
    and the incremental financial benefit from optimised charging.

    Assumes planned charging deviations are physically delivered.
    Consequently, the original forecast error carries through to
    the optimised demand schedule.

    Args:
        optimised_dataset: Settlement-period dataset containing original
            demand, optimised demand, planned deviations and prices.

    Returns:
        Tuple containing:
            - DataFrame with baseline and optimised financial metrics.
            - Dictionary containing aggregate financial performance.
    """

    df = optimised_dataset.copy()

    # Baseline procurement and settlement costs
    df["baseline_error_mwh"] = (
        df["actual_demand_mwh"] - df["forecast_demand_mwh"]
    )

    df["baseline_cost"] = (
        df["forecast_demand_mwh"] * df["day_ahead_price"]
    )

    df["baseline_error_cost"] = (
        df["baseline_error_mwh"] * df["system_price"]
    )

    df["baseline_total_cost"] = (
        df["baseline_cost"] + df["baseline_error_cost"]
    )

    # Apply planned deviations to realised charging demand
    df["optimised_actual_demand_mwh"] = (
        df["actual_demand_mwh"] + df["planned_deviation_mwh"]
    )

    # Calculate optimised procurement and settlement costs
    df["optimised_error_mwh"] = (
        df["optimised_actual_demand_mwh"]
        - df["optimised_demand_mwh"]
    )

    df["optimised_cost"] = (
        df["optimised_demand_mwh"] * df["day_ahead_price"]
    )

    df["optimised_error_cost"] = (
        df["optimised_error_mwh"] * df["system_price"]
    )

    df["optimised_total_cost"] = (
        df["optimised_cost"] + df["optimised_error_cost"]
    )

    # Aggregate financial performance
    baseline_total_cost = float(df["baseline_total_cost"].sum())
    optimised_total_cost = float(df["optimised_total_cost"].sum())

    gross_flex_value = float(
        (df["baseline_cost"] - df["optimised_cost"]).sum()
    )

    optimised_pnl_benefit = (
        baseline_total_cost - optimised_total_cost
    )

    shifted_mwh = float(
        df["planned_deviation_mwh"].clip(lower=0).sum()
    )

    financial_metrics = {
        "baseline_total_cost": baseline_total_cost,
        "optimised_total_cost": optimised_total_cost,
        "gross_flex_value": gross_flex_value,
        "optimised_pnl_benefit": optimised_pnl_benefit,
        "shifted_mwh": shifted_mwh,
    }

    return df, financial_metrics


def run_flex_sensitivity(
    core_dataset: pd.DataFrame,
    fleet_config: FleetConfig,
    charging_config: ChargingConfig,
    flex_levels: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
    method: str = "highs",
) -> tuple[pd.DataFrame, dict[float, dict]]:
    """
    Evaluate the financial value of different charging flexibility levels.

    Runs the charging optimiser for each flexibility scenario while
    keeping the fleet configuration and charging windows unchanged.

    Args:
        core_dataset: Prepared settlement-period dataset.
        fleet_config: Operator-managed fleet configuration.
        charging_config: Customer-managed charging preferences.
        flex_levels: Flexibility percentages to evaluate.
        method: Linear programming solver.

    Returns:
        Tuple containing:
            - DataFrame comparing financial outcomes across scenarios.
            - Dictionary containing detailed results for each scenario.

    Raises:
        ValueError: If a scenario exceeds operator flexibility limits.
    """

    levels = [float(level) for level in flex_levels]

    if not levels or any(
        not np.isfinite(level)
        or not 0 <= level <= fleet_config.max_flex_level
        for level in levels
    ):
        raise ValueError(
            "Flexibility scenarios must be within operator limits."
        )

    results = []
    scenario_outputs = {}

    # Forecast risk is independent of the flexibility level under
    # the current assumption that baseline forecast errors persist
    risk_analysis = analyse_forecast_risk(core_dataset)

    # Evaluate flexibility scenarios in ascending order
    for flex_level in sorted(set(levels)):

        # Preserve customer charging times while varying flexibility
        scenario_config = replace(
            charging_config,
            flex_level=flex_level,
        )

        # Run the charging optimiser
        optimised_dataset, window_results = run_optimiser(
            core_dataset=core_dataset,
            charging_config=scenario_config,
            fleet_config=fleet_config,
            method=method,
        )

        # Calculate financial performance
        financial_dataset, financial_metrics = (
            calculate_financial_performance(optimised_dataset)
        )

        # Store scenario-level financial metrics
        results.append({
            "flex_level": flex_level,
            **financial_metrics,
        })

        # Retain detailed results for each flexibility scenario
        scenario_outputs[flex_level] = {
            "dataset": financial_dataset,
            "window_results": window_results,
            "financial_metrics": financial_metrics,
            "risk_metrics": risk_analysis["risk_metrics"],
            "error_symmetry": risk_analysis["error_symmetry"],
            "error_by_sp": risk_analysis["error_by_sp"],
            "error_drivers": risk_analysis["error_drivers"],
        }

    # Consolidate flexibility scenario results
    flex_results = pd.DataFrame(results)

    # Calculate marginal financial value between flexibility levels
    flex_results["marginal_flex_value"] = (
        flex_results["gross_flex_value"].diff()
    )

    return flex_results, scenario_outputs


if __name__ == "__main__":

    # Initialise operator and customer configurations
    fleet_config = FleetConfig()
    charging_config = ChargingConfig()

    # Load and prepare input data
    prices, fleet_demand = load_data()

    core_dataset = prepare_data(
        prices,
        fleet_demand,
        max_fleet_capacity_mwh=fleet_config.max_fleet_capacity_mwh,
    )

    # Evaluate baseline forecasting risk
    risk_analysis = analyse_forecast_risk(core_dataset)

    print("\nFORECAST RISK METRICS")
    print(pd.Series(risk_analysis["risk_metrics"]).to_string())

    print("\nFORECAST ERROR DIRECTION")
    print(risk_analysis["error_symmetry"].to_string())

    # Run flexibility sensitivity analysis
    flex_results, scenario_outputs = run_flex_sensitivity(
        core_dataset=core_dataset,
        fleet_config=fleet_config,
        charging_config=charging_config,
    )

    print("\nFLEXIBILITY SENSITIVITY ANALYSIS")
    print(
        flex_results[
            [
                "flex_level",
                "gross_flex_value",
                "optimised_pnl_benefit",
                "marginal_flex_value",
                "shifted_mwh",
            ]
        ].to_string(index=False)
    )
