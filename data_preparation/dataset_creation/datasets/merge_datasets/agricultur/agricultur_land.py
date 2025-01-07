import pandas as pd
from functools import reduce


df = pd.read_csv('../../datasets/agricultur/land_use_area.csv')
df_share_area = pd.read_csv('../../datasets/agricultur/land_use_share_area.csv')
df_share_agriculture = pd.read_csv('../../datasets/agricultur/land_use_share_agriculture.csv')
df_share_cropland = pd.read_csv('../../datasets/agricultur/land_use_share_cropland.csv')

# Видалення зайвих колонок
columns_to_drop = [
    "Domain Code", "Domain", "Area Code (M49)", "Element Code",
    "Item Code", "Year Code", "Flag", "Flag Description", "Note", "Unit"
]
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned.rename(columns={df_cleaned.columns[0]: 'Country'}, inplace=True)

df_share_area = df_share_area.drop(columns=columns_to_drop)
df_share_area.rename(columns={df_share_area.columns[0]: 'Country'}, inplace=True)

df_share_agriculture = df_share_agriculture.drop(columns=columns_to_drop)
df_share_agriculture.rename(columns={df_share_agriculture.columns[0]: 'Country'}, inplace=True)

df_share_cropland = df_share_cropland.drop(columns=columns_to_drop)
df_share_cropland.rename(columns={df_share_cropland.columns[0]: 'Country'}, inplace=True)

# Додавання суфікса до назв колонок
df_cleaned["Item"] = df_cleaned["Item"].str.replace(" ", "_") + "_area_1000ha"
df_share_area["Item"] = df_share_area["Item"].str.replace(" ", "_") + "share_area_%"
df_share_agriculture["Item"] = df_share_agriculture["Item"].str.replace(" ", "_") + "share_agriculture_%"

# Перехід у широкий формат
df_wide = df_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()
df_share_area = df_share_area.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()
df_share_agriculture = df_share_agriculture.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()

print(df_wide.head())
print(df_share_area.head())
print(df_share_agriculture.head())

df_wide = df_wide.drop(columns="Farm_buildings_and_Farmyards_area_1000ha")

df_wide.to_csv('datasets/land_use_area.csv', index=False)
df_share_area.to_csv('datasets/land_use_area_share.csv', index=False)
df_share_agriculture.to_csv('datasets/land_use_agriculture_share.csv', index=False)

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [df_wide, df_share_area, df_share_agriculture]
)

merged_data.to_csv('datasets/land_use.csv', index=False)
