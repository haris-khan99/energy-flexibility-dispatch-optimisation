
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import FleetConfig, ChargingConfig


def optimise_energy(
    charging_window: pd.DataFrame,
    charging_config: ChargingConfig,
    method: str = "highs",
) -> dict:
    """
    Optimise EV charging demand within a specified charging window.

    Minimises day-ahead electricity costs subject to energy conservation,
    customer flexibility preferences and physical charging capacity.

    Args:
        charging_window: Settlement-period observations within the
            configured charging window.
        charging_config: Customer-defined charging preferences, including
            start time, completion deadline and flexibility level.
        method: Linear programming solver used by scipy.optimize.linprog.

    Raises:
        ValueError: If inputs or optimisation constraints are invalid.
        RuntimeError: If the linear programming solver fails.

    Returns:
        Dictionary containing optimised allocations, planned deviations,
        procurement costs and flexibility value metrics.
    """

    if charging_window.empty:
        raise ValueError("Cannot optimise an empty charging window.")

    flex_level = charging_config.flex_level

    if not 0 <= flex_level <= 1:
        raise ValueError("Flex level must be between 0 and 1.")

    baseline = charging_window["baseline_demand_mwh"].to_numpy(dtype=float)
    prices = charging_window["day_ahead_price"].to_numpy(dtype=float)
    capacity = charging_window["physical_capacity_mwh"].to_numpy(dtype=float)

    if not all(np.isfinite(x).all() for x in (baseline, prices, capacity)):
        raise ValueError("Optimisation inputs must contain finite values.")

    if np.any(baseline < 0) or np.any(capacity < 0):
        raise ValueError("Baseline demand and capacity cannot be negative.")

    # Establish flexibility and physical capacity constraints
    lower_bounds = np.maximum(baseline * (1 - flex_level), 0)
    upper_bounds = np.minimum(baseline * (1 + flex_level), capacity)

    if np.any(lower_bounds > upper_bounds):
        raise ValueError(
            "Infeasible constraints: minimum charging demand exceeds "
            "available physical capacity."
        )

    # Preserve total energy across the charging window
    total_energy = baseline.sum()

    if not (
        lower_bounds.sum() - 1e-8
        <= total_energy
        <= upper_bounds.sum() + 1e-8
    ):
        raise ValueError(
            "Infeasible constraints: required energy cannot be delivered "
            "within the charging window."
        )

    bounds = list(zip(lower_bounds, upper_bounds))

    # Solve the linear programming problem
    result = linprog(
        c=prices,
        A_eq=np.ones((1, len(baseline))),
        b_eq=np.array([total_energy]),
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise RuntimeError(
            f"Linear programming failed: {result.message}"
        )

    # Calculate optimised allocation and deviations
    optimised_allocation = result.x
    planned_deviation_mwh = optimised_allocation - baseline

    # Calculate day-ahead procurement costs
    baseline_cost = float(np.dot(prices, baseline))
    optimised_cost = float(np.dot(prices, optimised_allocation))

    gross_flex_value = baseline_cost - optimised_cost

    # Count upward shifts only to avoid double-counting moved energy
    shifted_mwh = float(np.maximum(planned_deviation_mwh, 0).sum())

    return {
        "window_start": charging_window["datetime"].min(),
        "window_end": charging_window["datetime"].max(),
        "flex_fraction": flex_level,
        "baseline": baseline,
        "prices": prices,
        "optimised_allocation": optimised_allocation,
        "planned_deviation_mwh": planned_deviation_mwh,
        "baseline_cost": baseline_cost,
        "optimised_cost": optimised_cost,
        "gross_flex_value": gross_flex_value,
        "shifted": shifted_mwh,
    }


def create_charging_windows(
    dataset: pd.DataFrame,
    charging_config: ChargingConfig,
) -> list[pd.DataFrame]:
    """
    Partition settlement-period data into customer-defined charging windows.

    Supports overnight charging windows that cross midnight. Each window
    is identified by its charging start date.

    Args:
        dataset: Prepared settlement-period dataset.
        charging_config: Customer-defined start time and completion deadline.

    Returns:
        List of DataFrames, each representing one complete charging window.
    """

    start_time = charging_config.start_time
    end_time = charging_config.end_time

    if start_time == end_time:
        raise ValueError(
            "Charging start and end times must differ."
        )

    dataset = dataset.copy()
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])

    dataset = dataset.sort_values("datetime").reset_index(drop=True)

    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute

    observation_minutes = (
        dataset["datetime"].dt.hour * 60
        + dataset["datetime"].dt.minute
    )

    overnight = end_minutes < start_minutes

    if overnight:
        in_window = (
            (observation_minutes >= start_minutes)
            | (observation_minutes < end_minutes)
        )
    else:
        in_window = (
            (observation_minutes >= start_minutes)
            & (observation_minutes < end_minutes)
        )

    dataset = dataset.loc[in_window].copy()

    # Associate post-midnight observations with the previous start date
    dataset["window_date"] = dataset["datetime"].dt.normalize()

    if overnight:
        after_midnight = (
            dataset["datetime"].dt.hour * 60
            + dataset["datetime"].dt.minute
        ) < end_minutes

        dataset.loc[after_midnight, "window_date"] -= pd.Timedelta(days=1)

    charging_windows = [
        window.copy()
        for _, window in dataset.groupby("window_date", sort=True)
    ]

    return charging_windows


def run_optimiser(core_dataset: pd.DataFrame, charging_config: ChargingConfig, fleet_config: FleetConfig, method: str = "highs") -> tuple[pd.DataFrame, list[dict]]:
    """
    Run the EV charging optimiser across customer-defined charging windows.

    Each window is optimised independently, allowing energy to be shifted
    across midnight while respecting customer preferences and physical
    fleet capacity.

    Args:
        core_dataset: Prepared dataset containing baseline demand,
            day-ahead prices and physical charging capacity.
        charging_config: Customer-defined charging preferences.
        fleet_config: Operator-managed fleet configuration and limits.
        method: Linear programming solver.

    Returns:
        Tuple containing:
            - Dataset with optimised demand and planned deviations.
            - List of charging-window optimisation results.
    """

    if not 0 <= charging_config.flex_level <= fleet_config.max_flex_level:
        raise ValueError(
            "Customer flexibility exceeds the operator's permitted limits."
        )

    optimised_dataset = core_dataset.copy()

    optimised_dataset["optimised_demand_mwh"] = np.nan
    optimised_dataset["planned_deviation_mwh"] = np.nan

    # Preserve original row indices for assigning optimisation results
    charging_windows = create_charging_windows(
        optimised_dataset,
        charging_config,
    )

    window_results = []

    for window in charging_windows:

        result = optimise_energy(
            charging_window=window,
            charging_config=charging_config,
            method=method,
        )

        # Match results to original settlement-period records
        mask = optimised_dataset.index.isin(window["_original_index"])

        optimised_dataset.loc[
            mask, "optimised_demand_mwh"
        ] = result["optimised_allocation"]

        optimised_dataset.loc[
            mask, "planned_deviation_mwh"
        ] = result["planned_deviation_mwh"]

        window_results.append(result)

    return optimised_dataset, window_results
