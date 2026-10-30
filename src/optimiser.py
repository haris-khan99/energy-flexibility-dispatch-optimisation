import numpy as np
import pandas as pd
from scipy.optimize import linprog
from config import flex_level, optimisation_method

def optimise_energy(data_by_date: pd.DataFrame, flex_level: float = flex_level, method: str = optimisation_method) -> dict:
    """
    Optimise daily EV charging demand to minimise day-ahead electricity costs.

    Uses linear programming to redistribute forecast charging demand across
    settlement periods, subject to energy conservation, flexibility limits,
    and physical fleet charging capacity.

    Args:
        data_by_date (pd.DataFrame): Daily settlement-period dataset containing forecast demand, day-ahead prices and physical charging capacity.
        flex_level (float, optional): Maximum fraction of baseline demand that can be shifted up or down within each settlement period.
        method (str, optional): Linear programming solver used by scipy.optimize.linprog.

    Raises:
        ValueError: If flexibility level is not between 0 and 1, or if lower bounds exceed upper bounds, or if total energy cannot be delivered within the bounds.
        RuntimeError: If the linear programming solver fails to find a solution.

    Returns:
        dict: Dictionary containing optimised allocations, planned deviations, baseline and optimised costs, and flexibility value metrics.
    """
    
    
    baseline = data_by_date['baseline_demand_mwh'].to_numpy(dtype=float)
    prices = data_by_date['day_ahead_price'].to_numpy(dtype=float)
    capacity = data_by_date['physical_capacity_mwh'].to_numpy(dtype=float)

    if not 0 < flex_level <= 1:
        raise ValueError("Flex level must be between 0 and 1.")

    lower_bounds = np.maximum(baseline * (1 - flex_level), 0)
    upper_bounds = np.minimum(baseline * (1 + flex_level), capacity)

    if np.any(lower_bounds > upper_bounds):
        raise ValueError("Lower bounds exceed upper bounds for some time periods. Check the flex level and baseline values.")

    total_energy = np.sum(baseline)

    if not (
        lower_bounds.sum() - 1e-8
        <= total_energy
        <= upper_bounds.sum() + 1e-8
    ):
        raise ValueError(
            "Infeasible constraints: forecast energy cannot be "
            "delivered within the daily charging bounds."
        )

    bounds = list(zip(lower_bounds, upper_bounds))

    result = linprog(
        c=prices,
        A_eq=np.ones((1, len(baseline))),
        b_eq=np.array([total_energy]),
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise RuntimeError(f"Linear programming failed: {result.message}")

    optimised_allocation = result.x

    planned_deviation_mwh = optimised_allocation - baseline

    baseline_cost = float(np.dot(prices, baseline))
    optimised_cost = float(np.dot(prices, optimised_allocation))

    gross_flex_value = baseline_cost - optimised_cost

    shifted_mwh = float(np.maximum(planned_deviation_mwh, 0).sum())

    return {
        "date": data_by_date["date"].iloc[0],
        "flex_fraction": flex_level,
        "baseline": baseline,
        "prices": prices,
        "optimised_allocation": optimised_allocation,
        "planned_deviation_mwh": planned_deviation_mwh,
        "baseline_cost": baseline_cost,
        "optimised_cost": optimised_cost,
        "gross_flex_value": gross_flex_value,
        "shifted": shifted_mwh
    }

