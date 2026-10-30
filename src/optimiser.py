"""EV charging optimisation over complete, configurable charging windows."""

from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from config import FleetConfig, ChargingConfig


def optimise_energy(
    charging_window: pd.DataFrame,
    charging_config: ChargingConfig,
    method: str = "highs",
    settlement_period_hours: float = 0.5,
) -> dict:
    """Minimise procurement costs while conserving charging-window energy."""
    if charging_window.empty:
        raise ValueError("Cannot optimise an empty charging window.")

    flex_level = charging_config.flex_level
    if not 0 <= flex_level <= 1:
        raise ValueError("Flex level must be between 0 and 1.")

    baseline = charging_window["forecast_demand_mwh"].to_numpy(dtype=float)
    prices = charging_window["day_ahead_price"].to_numpy(dtype=float)
    capacity = charging_window["physical_capacity_mwh"].to_numpy(dtype=float)

    if not all(np.isfinite(x).all() for x in (baseline, prices, capacity)):
        raise ValueError("Optimisation inputs must be finite.")
    if (baseline < 0).any() or (capacity < 0).any():
        raise ValueError("Demand and charging capacity cannot be negative.")

    lower = np.maximum(baseline * (1 - flex_level), 0)
    upper = np.minimum(baseline * (1 + flex_level), capacity)
    if (lower > upper + 1e-9).any():
        raise ValueError("Infeasible: minimum charging demand exceeds capacity.")

    total_energy = float(baseline.sum())
    if not lower.sum() - 1e-8 <= total_energy <= upper.sum() + 1e-8:
        raise ValueError("Infeasible: required energy exceeds charging-window bounds.")

    result = linprog(
        c=prices,
        A_eq=np.ones((1, len(baseline))),
        b_eq=[total_energy],
        bounds=list(zip(lower, upper)),
        method=method,
    )
    if not result.success:
        raise RuntimeError(f"Linear programming failed: {result.message}")

    allocation = result.x
    deviation = allocation - baseline
    baseline_cost = float(np.dot(prices, baseline))
    optimised_cost = float(np.dot(prices, allocation))

    return {
        "window_start": charging_window["datetime"].iloc[0],
        "window_end": charging_window["datetime"].iloc[-1]
        + pd.Timedelta(hours=settlement_period_hours),
        "flex_fraction": flex_level,
        "baseline": baseline,
        "prices": prices,
        "optimised_allocation": allocation,
        "planned_deviation_mwh": deviation,
        "baseline_cost": baseline_cost,
        "optimised_cost": optimised_cost,
        "gross_flex_value": baseline_cost - optimised_cost,
        "shifted": float(np.maximum(deviation, 0).sum()),
    }


def create_charging_windows(
    dataset: pd.DataFrame,
    charging_config: ChargingConfig,
    settlement_period_hours: float = 0.5,
) -> list[pd.DataFrame]:
    """Collect complete charging windows, including those crossing midnight."""
    start_time = charging_config.start_time
    end_time = charging_config.end_time
    if start_time == end_time:
        raise ValueError("Charging start and end times must differ.")

    period_minutes = settlement_period_hours * 60
    if period_minutes <= 0 or not float(period_minutes).is_integer():
        raise ValueError("Settlement period must be a positive whole number of minutes.")
    period_minutes = int(period_minutes)
    if (24 * 60) % period_minutes:
        raise ValueError("Settlement period must divide evenly into 24 hours.")

    def clock_minutes(value):
        if value.second or value.microsecond:
            raise ValueError("Charging times must align with settlement periods.")
        minute = value.hour * 60 + value.minute
        if minute % period_minutes:
            raise ValueError("Charging times must align with settlement periods.")
        return minute

    start_minutes = clock_minutes(start_time)
    end_minutes = clock_minutes(end_time)
    overnight = end_minutes < start_minutes

    df = dataset.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].isna().any():
        raise ValueError("datetime contains missing values.")
    df = df.sort_values("datetime")  # Preserve input row indices

    minutes = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    if overnight:
        in_window = (minutes >= start_minutes) | (minutes < end_minutes)
    else:
        in_window = (minutes >= start_minutes) & (minutes < end_minutes)
    df = df.loc[in_window].copy()

    # Local calendar dates (without timezone offsets) identify a window's start day.
    df["window_date"] = df["datetime"].dt.tz_localize(None).dt.normalize()
    if overnight:
        after_midnight = minutes.loc[df.index] < end_minutes
        df.loc[after_midnight, "window_date"] -= pd.Timedelta(days=1)

    grouped = list(df.groupby("window_date", sort=True))
    windows = []
    tz = df["datetime"].dt.tz
    for i, (window_date, window) in enumerate(grouped):
        start = pd.Timestamp.combine(window_date.date(), start_time)
        end_date = window_date.date() + timedelta(days=int(overnight))
        end = pd.Timestamp.combine(end_date, end_time)
        if tz is not None:
            start = start.tz_localize(tz)
            end = end.tz_localize(tz)

        expected = pd.date_range(
            start=start,
            end=end,
            freq=pd.Timedelta(minutes=period_minutes),
            inclusive="left",
        )
        actual = pd.DatetimeIndex(window["datetime"])
        if not actual.equals(expected):
            if i == 0 or i == len(grouped) - 1:
                warnings.warn(
                    f"Skipping incomplete boundary window starting {window_date.date()} "
                    f"({len(actual)} observed vs {len(expected)} expected periods).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise ValueError(
                f"Incomplete or duplicated settlement periods in window "
                f"starting {window_date.date()}."
            )

        window = window.copy()
        windows.append(window)

    return windows


def run_optimiser(
    core_dataset: pd.DataFrame,
    charging_config: ChargingConfig,
    fleet_config: FleetConfig,
    method: str = "highs",
) -> tuple[pd.DataFrame, list[dict]]:
    """Optimise complete windows and preserve the baseline outside those windows."""
    if not 0 <= charging_config.flex_level <= fleet_config.max_flex_level <= 1:
        raise ValueError("Customer flexibility must be within operator limits.")

    optimised = core_dataset.copy().reset_index(drop=True)
    optimised["datetime"] = pd.to_datetime(optimised["datetime"])
    optimised["optimised_demand_mwh"] = optimised["forecast_demand_mwh"].astype(float)
    optimised["planned_deviation_mwh"] = 0.0

    windows = create_charging_windows(
        optimised, charging_config, fleet_config.settlement_period_hours
    )
    if not windows:
        raise ValueError("No complete charging windows were found in the dataset.")

    results = []
    for window in windows:
        result = optimise_energy(
            window, charging_config, method, fleet_config.settlement_period_hours
        )
        optimised.loc[window.index, "optimised_demand_mwh"] = result["optimised_allocation"]
        optimised.loc[window.index, "planned_deviation_mwh"] = result["planned_deviation_mwh"]
        results.append(result)

    return optimised, results


if __name__ == "__main__":
    from data_processor import load_data, prepare_data

    fleet_config = FleetConfig()
    charging_config = ChargingConfig()

    prices, fleet_demand = load_data()
    core_dataset = prepare_data(
        prices,
        fleet_demand,
        max_fleet_capacity_mwh=fleet_config.max_fleet_capacity_mwh,
    )

    optimised_dataset, window_results = run_optimiser(
        core_dataset=core_dataset,
        charging_config=charging_config,
        fleet_config=fleet_config,
    )

    print(optimised_dataset.head())
    print(f"\nCharging windows optimised: {len(window_results)}")
    print(
        "Total gross flex value: "
        f"£{sum(r['gross_flex_value'] for r in window_results):,.2f}"
    )
