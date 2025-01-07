import pandas as pd
from functools import reduce


land_use = pd.read_csv('datasets/land_use.csv')
fertilize_use = pd.read_csv('datasets/fertilize.csv')
pesticides_use = pd.read_csv('datasets/pesticides_use.csv')

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [land_use, fertilize_use, pesticides_use]
)

merged_data.to_csv('datasets/agriculture_features.csv', index=False)
