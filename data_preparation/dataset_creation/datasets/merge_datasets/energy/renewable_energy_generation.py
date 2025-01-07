import pandas as pd
from functools import reduce

nuclear = pd.read_csv('../../datasets/renewable_energy/generation/nuclear_generation.csv')
hydro = pd.read_csv('../../datasets/renewable_energy/generation/hydro_generation.csv')
solar = pd.read_csv('../../datasets/renewable_energy/generation/solar_generation.csv')
wind = pd.read_csv('../../datasets/renewable_energy/generation/wind_generation.csv')

name = 'Energy Consumption'

nuclear.rename(columns={name: "Nuclear_TWh"}, inplace=True)
hydro.rename(columns={name: "Hydro_TWh"}, inplace=True)
solar.rename(columns={name: "Solar_TWh"}, inplace=True)
wind.rename(columns={name: "Wind_TWh"}, inplace=True)

df = [nuclear, hydro, solar, wind]

# Об'єднання за допомогою reduce
merged_data = reduce(lambda left, right: pd.merge(left, right, on=["Country", "Year"],
                                                  how="outer", validate='one_to_one'), df)

print(merged_data.head())

merged_data.to_csv("datasets/renewable_energy_combined.csv", index=False)

