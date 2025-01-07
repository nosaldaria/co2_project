import pandas as pd
from functools import reduce


df = pd.read_csv('../../datasets/agricultur/fertilizer_use.csv')
df_per = pd.read_csv('../../datasets/agricultur/fertilizer_use_per_agriculture_product.csv')

# Видалення зайвих колонок
columns_to_drop = [
    "Domain Code", "Domain", "Area Code (M49)", "Element Code",
    "Item Code", "Year Code", "Flag", "Flag Description", "Note", "Unit"
]
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned.rename(columns={df_cleaned.columns[0]: 'Country'}, inplace=True)

df_per_cleaned = df_per.drop(columns=columns_to_drop)
df_per_cleaned.rename(columns={df_per_cleaned.columns[0]: 'Country'}, inplace=True)

# Додавання суфікса до назв колонок
df_cleaned["Item"] = df_cleaned["Item"].str.replace(" ", "_") + "_t"
df_per_cleaned["Item"] = df_per_cleaned["Item"].str.replace(" ", "_") + "_g/Int$"

# Перехід у широкий формат
df_wide = df_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()
df_wide_per = df_per_cleaned.pivot(index=["Year", "Country"], columns="Item", values="Value").reset_index()

print(df_wide.head())

df_wide.to_csv('datasets/fertilizer_use_agriculture.csv', index=False)
df_wide_per.to_csv('datasets/fertilizer_use_per_agriculture_products.csv', index=False)

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [df_wide, df_wide_per]
)

merged_data.to_csv('datasets/fertilize.csv', index=False)
