import pandas as pd
from functools import reduce


df = pd.read_csv('../../datasets/forest/forest_area.csv')
df_per_land = pd.read_csv('../../datasets/forest/forest_share_area_land.csv')
df_per_forest_land = pd.read_csv('../../datasets/forest/forest_share_forest_land.csv')

# Видалення зайвих колонок
columns_to_drop = [
    "Domain Code", "Domain", "Area Code (M49)", "Element Code",
    "Item Code", "Year Code", "Flag", "Flag Description", "Note", "Unit"
]
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned.rename(columns={df_cleaned.columns[0]: 'Country'}, inplace=True)

df_per_land_cleaned = df_per_land.drop(columns=columns_to_drop)
df_per_land_cleaned.rename(columns={df_per_land_cleaned.columns[0]: 'Country'}, inplace=True)

df_per_forest_cleaned = df_per_forest_land.drop(columns=columns_to_drop)
df_per_forest_cleaned.rename(columns={df_per_forest_cleaned.columns[0]: 'Country'}, inplace=True)

# Додавання суфікса до назв колонок
df_cleaned["Item"] = df_cleaned["Item"].str.replace(" ", "_") + "_area_1000_ha"
df_per_land_cleaned["Item"] = df_per_land_cleaned["Item"].str.replace(" ", "_") + "_share_land_area_%"
df_per_forest_cleaned["Item"] = df_per_forest_cleaned["Item"].str.replace(" ", "_") + "_share_forest_land_%"

# Перехід у широкий формат
df_wide = df_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()
df_per_land_cleaned = df_per_land_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()
df_per_forest_cleaned = df_per_forest_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()

df_wide.to_csv('datasets/forest_area.csv', index=False)
df_per_land_cleaned.to_csv('datasets/forest_share_land_area.csv', index=False)
df_per_forest_cleaned.to_csv('datasets/forest_share_forest_land.csv', index=False)

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [df_wide, df_per_land_cleaned, df_per_forest_cleaned]
)

merged_data.to_csv('datasets/forest_data.csv', index=False)
