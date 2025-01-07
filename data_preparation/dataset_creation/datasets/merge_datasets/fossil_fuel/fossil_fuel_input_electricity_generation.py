import pandas as pd
from functools import reduce


oil = pd.read_csv('../../datasets/fossil_fuels/input_electricity/Oil_inputs_electricity_generation_ej.csv')
gas = pd.read_csv('../../datasets/fossil_fuels/input_electricity/Gas_inputs_electricity_generation_ej.csv')
coal = pd.read_csv('../../datasets/fossil_fuels/input_electricity/Coal_inputs_electricity_generation_ej.csv')

name = 'Energy Consumption'

oil.rename(columns={name: "Oil_Electricity"}, inplace=True)
gas.rename(columns={name: "Gas_Electricity"}, inplace=True)
coal.rename(columns={name: "Coal_Electricity"}, inplace=True)

fossil_electricity = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="inner", validate='one_to_one'),
    [oil, gas, coal]
)

print(fossil_electricity.head())

fossil_electricity.to_csv("datasets/fossil_electricity_combined.csv", index=False)

