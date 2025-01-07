import pandas as pd
from functools import reduce


oil = pd.read_csv('../../datasets/fossil_fuels/production/Oil_production_tonnes.csv')
gas = pd.read_csv('../../datasets/fossil_fuels/production/Gas_production_ej.csv')
coal = pd.read_csv('../../datasets/fossil_fuels/production/Coal_production_ej.csv')

name = 'Energy Consumption'

oil.rename(columns={name: "Oil_Production"}, inplace=True)
gas.rename(columns={name: "Gas_Production"}, inplace=True)
coal.rename(columns={name: "Coal_Production"}, inplace=True)

fossil_fuels = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [oil, gas, coal]
)

print(fossil_fuels.head())

fossil_fuels.to_csv("datasets/fossil_fuels_combined.csv", index=False)

