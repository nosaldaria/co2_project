import pandas as pd
import numpy as np


df = pd.read_csv('datasets/final_dataset.csv')

# Видалення непотрібних колонок
columns_to_drop = [
    'Annual CO₂ emissions per GDP (kg per international-$)',
    'Annual CO₂ emissions (per capita)',
    'Cumulative CO₂ emissions'
]
data = df.drop(columns=columns_to_drop)

# Групування за країною та роком
grouped_data = data.groupby(['Country', 'Year'], as_index=False).agg({
    'Coal Consumption': 'sum',
    'Gas Consumption': 'sum',
    'Oil Consumption': 'sum',
    'Hydro Consumption': 'sum',
    'Solar Consumption': 'sum',
    'Wind Consumption': 'sum',
    'Nuclear Consumption': 'sum',
    'Primary Energy Consumption': 'sum',
    'Annual CO₂ emissions': 'sum'
})

# Обчислення Fossil Share, залишаючи NaN там, де є пропуски або Primary Energy Consumption == 0
grouped_data['Fossil Share Consumption'] = np.where(
    grouped_data['Primary Energy Consumption'] > 0,
    (grouped_data['Coal Consumption'] +
     grouped_data['Gas Consumption'] +
     grouped_data['Oil Consumption']) / grouped_data['Primary Energy Consumption'],
    np.nan
)

grouped_data['Renewable Share Consumption'] = np.where(
    grouped_data['Primary Energy Consumption'] > 0,
    (grouped_data['Hydro Consumption'] +
     grouped_data['Solar Consumption'] +
     grouped_data['Wind Consumption']) / grouped_data['Primary Energy Consumption'],
    np.nan
)

grouped_data['Nuclear Share Consumption'] = np.where(
    grouped_data['Primary Energy Consumption'] > 0,
    grouped_data['Nuclear Consumption'] / grouped_data['Primary Energy Consumption'],
    np.nan
)

grouped_data['Non-CO₂ Share Consumption'] = np.where(
    grouped_data['Primary Energy Consumption'] > 0,
    (grouped_data['Hydro Consumption'] +
     grouped_data['Solar Consumption'] +
     grouped_data['Wind Consumption'] +
     grouped_data['Nuclear Consumption']) / grouped_data['Primary Energy Consumption'],
    np.nan
)

print(grouped_data.head())

# Забезпечуємо, що дубльовані колонки не додаються:
# Вибираємо тільки ті колонки, які відсутні в оригінальному датасеті
columns_to_add = grouped_data.columns.difference(data.columns).to_list()
final_data = pd.concat([data, grouped_data[columns_to_add]], axis=1)

# Видалення непотрібних колонок
columns_to_drop = [
    'Coal Consumption',
    'Gas Consumption',
    'Hydro Consumption',
    'Nuclear Consumption',
    'Oil Consumption',
    'Solar Consumption',
    'Wind Consumption'
]
final_data = final_data.drop(columns=columns_to_drop)

print(final_data.head())

final_data.to_csv("datasets/co2_drops.csv", index=False)
