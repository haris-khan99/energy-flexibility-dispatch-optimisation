import pandas as pd
from config import max_fleet_capacity_mwh, prices_path, fleet_demand_path
from pathlib import Path

def load_data(prices_path: Path = prices_path, fleet_demand_path: Path = fleet_demand_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load electricity price and EV fleet demand data from CSV files.

    Args:
        prices_path (Path, optional): Path to the electricity price dataset.
        fleet_demand_path (Path, optional): Path to the EV fleet demand dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the price and fleet demand DataFrames.
    """

    prices = pd.read_csv(prices_path, parse_dates=['datetime'])
    fleet_demand = pd.read_csv(fleet_demand_path, parse_dates=['datetime'])

    return prices, fleet_demand

def prepare_data(prices: pd.DataFrame, fleet_demand: pd.DataFrame, max_fleet_capacity_mwh: float = max_fleet_capacity_mwh) -> pd.DataFrame:
    """
    Prepare the combined dataset for EV fleet optimisation.

    Merges electricity prices with fleet demand, calculates historical
    charger connectivity and derives available physical charging capacity.    

    Args:
        prices (pd.DataFrame): Electricity price DataFrame.
        fleet_demand (pd.DataFrame): EV fleet demand DataFrame.
        max_fleet_capacity_mwh (float, optional): Maximum theoretical fleet capacity in MWh.

    Returns:
        pd.DataFrame: Prepared DataFrame containing the features required for optimisation.
    """

    core_dataset = pd.merge(
        prices,
        fleet_demand,
        how='inner',
        on=['datetime', 'settlement_period']
        )

    core_dataset = (core_dataset
        .sort_values(["datetime", "settlement_period"])
        .reset_index(drop=True))

    core_dataset.insert(1, 'date', core_dataset['datetime'].dt.date)
 
    # To mitigrate data leakage: calculate a historical expanding average of the fraction of plugged in chargers by settlement period
    # as a result the first day of data is dropped from the dataset to avoid look-ahead bias
    core_dataset['plugged_in_fraction_avg'] = (
        core_dataset
            .groupby('settlement_period')['plugged_in_fraction']
            .transform(
                lambda x: x.shift(1).expanding().mean()
            )
    )

    core_dataset = (
        core_dataset
            .dropna(subset=['plugged_in_fraction_avg'])
            .reset_index(drop=True)
    )

    # Considering the 10,000 EV chargers mentioned in the doc, the usuable physical capacity is determined by the 
    # maximum of 35 MWh per half hour multipled by the fraction of the fleet connected
    core_dataset['physical_capacity_mwh'] = max_fleet_capacity_mwh * core_dataset['plugged_in_fraction_avg']

    return core_dataset

if __name__ == "__main__":
    prices, fleet_demand = load_data()
    prepared_data = prepare_data(prices, fleet_demand)
    print(prepared_data.head())