# config.py

from pathlib import Path

project_root = Path(__file__).parent.parent

prices_path = project_root / 'prices.csv'
fleet_demand_path = project_root / 'fleet_demand.csv'

# EV fleet parameters
n_chargers = 10_000
charger_power_kw = 7
settlement_period_hours = 0.5

# Maximum fleet capacity per settlement period (MWh)
max_fleet_capacity_mwh = (
    n_chargers
    * charger_power_kw
    * settlement_period_hours
    / 1000
)

flex_level = 0.5  # Default flexibility level (50%)
optimisation_method = 'highs'  # Default optimisation method