# config.py

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