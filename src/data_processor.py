import pandas as pd
from config import max_fleet_capacity_mwh

def data_preparation():
    """_summary_

    Returns:
        _type_: _description_
    """


    prices = pd.read_csv('prices.csv', parse_dates=['datetime'])
    fleet_demand = pd.read_csv('fleet_demand.csv', parse_dates=['datetime'])

    core_dataset = pd.merge(prices, fleet_demand, how='inner', on=['datetime', 'settlement_period'])
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