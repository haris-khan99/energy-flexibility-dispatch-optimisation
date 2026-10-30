# config.py

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

project_root = Path(__file__).parent.parent

prices_path = project_root / 'prices.csv'
fleet_demand_path = project_root / 'fleet_demand.csv'

@dataclass(frozen=True)
class FleetConfig:
    """Operator-managed physical and operational configuration."""
    
    n_chargers: int = 10_000
    charger_power_kw: float = 7.0
    settlement_period_hours: float = 0.5
    max_flex_level : float = 0.5  # Maximum flexibility level (50%)
    optimisation_method = 'highs'  # Default optimisation method

    @property
    def max_fleet_capacity_mwh(self) -> float:
        """Maximum fleet capacity per settlement period (MWh)."""
        return (
            self.n_chargers
            * self.charger_power_kw
            * self.settlement_period_hours
            / 1000
        )

@dataclass(frozen=True)
class ChargingConfig:
    """Customer-managed charging preferences."""

    start_time: time = time(18, 0)  # Default start time for charging (6 PM)
    end_time: time = time(6, 0)  # Default end time for charging (6 AM)
    flex_level: float = 0.3  # Default flexibility level (50%)