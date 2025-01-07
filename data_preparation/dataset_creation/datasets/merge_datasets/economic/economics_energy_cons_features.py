import pandas as pd
import numpy as np


energy = pd.read_csv("../energy/datasets/combined_energy_consumption.csv")
economic = pd.read_csv("datasets/economics_features_new.csv")

print(f'{energy.head()}')
print(f'{economic.head()}')

merged_data = pd.merge(energy, economic, on=['Country', 'Year'], how='outer', validate="one_to_one")

print(merged_data.head())

merged_data.to_csv("datasets/merged_dataset_new.csv", index=False)

economic.to_csv("datasets/economics_features_new.csv", index=False)
