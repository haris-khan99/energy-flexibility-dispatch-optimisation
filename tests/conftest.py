"""Shared synthetic settlement-period data; no production CSV files required."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# The application uses bare imports (e.g. `from optimiser import ...`).
# Make those modules importable when pytest runs from the repo root.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from src.config import ChargingConfig, FleetConfig


@pytest.fixture
def fleet_config():
    return FleetConfig()


@pytest.fixture
def charging_config():
    return ChargingConfig()


@pytest.fixture
def core_dataset():
    """Two complete 18:00-06:00 sessions, with a daytime gap between them."""
    timestamps = pd.date_range(
        "2026-01-05 18:00", "2026-01-07 06:00",
        freq="30min", inclusive="left",
    )
    n = len(timestamps)
    hours = timestamps.hour + timestamps.minute / 60

    # Make evening charging costly and late-night charging cheaper.
    prices = np.where((hours >= 18) & (hours < 22), 150.0, 30.0)
    prices = np.where((hours >= 6) & (hours < 18), 80.0, prices)
    baseline = np.full(n, 10.0)
    actual = baseline + np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    return pd.DataFrame({
        "datetime": timestamps,
        "settlement_period": (timestamps.hour * 2 + timestamps.minute // 30 + 1),
        "forecast_demand_mwh": baseline,
        "actual_demand_mwh": actual,
        "day_ahead_price": prices,
        "system_price": np.full(n, 70.0),
        "physical_capacity_mwh": np.full(n, 20.0),
        "plugged_in_fraction": np.linspace(0.6, 0.9, n),
    })
