import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('datasets/electricity_generation_share.csv')

# Групування даних за країною та роком
grouped_land = df.groupby(['Country', 'Year'], as_index=False).agg({
    'Other_land_area_1000ha': 'sum',
    'Land_area_area_1000ha': 'sum',
    'Temporary_crops_area_1000ha': 'sum',
    'Permanent_crops_area_1000ha': 'sum',
    'Temporary_fallow_area_1000ha': 'sum',
    'Agricultural_land_area_1000ha': 'sum',
    'Perm._meadows_&_pastures_-_Cultivated_area_1000ha': 'sum',
    'Perm._meadows_&_pastures_-_Nat._growing_area_1000ha': 'sum',
    'Permanent_meadows_and_pastures_area_1000ha': 'sum'
})

# Обчислення частки сільськогосподарських земель від загальної площі
grouped_land['Temporary_crops_area_agriculture_share'] = np.where(
    grouped_land['Agricultural_land_area_1000ha'] > 0,
    grouped_land['Temporary_crops_area_1000ha'] / grouped_land['Agricultural_land_area_1000ha'],
    np.nan
)

# Обчислення частки інших земель від загальної площі
grouped_land['Other_land_share'] = np.where(
    grouped_land['Land_area_area_1000ha'] > 0,
    grouped_land['Other_land_area_1000ha'] / grouped_land['Land_area_area_1000ha'],
    np.nan
)

# Перевірка, які нові колонки додаються
columns_to_add = grouped_land.columns.difference(df.columns).to_list()

# Додавання нових колонок до оригінального DataFrame
final_data = pd.concat([df, grouped_land[columns_to_add]], axis=1)

# Групування даних за країною та роком
grouped_fertilizers_t = final_data.groupby(['Country', 'Year'], as_index=False).agg({
    'Nutrient_nitrogen_N_(total)_t': 'sum',
    'Nutrient_phosphate_P2O5_(total)_t': 'sum',
    'Nutrient_potash_K2O_(total)_t': 'sum'
})

# Обчислення загального обсягу добрив у тоннах
grouped_fertilizers_t['Total_Fertilizers_t'] = (
    grouped_fertilizers_t['Nutrient_nitrogen_N_(total)_t'] +
    grouped_fertilizers_t['Nutrient_phosphate_P2O5_(total)_t'] +
    grouped_fertilizers_t['Nutrient_potash_K2O_(total)_t']
)

# Обчислення частки кожного добрива
grouped_fertilizers_t['Nitrogen_Share_t'] = np.where(
    grouped_fertilizers_t['Total_Fertilizers_t'] > 0,
    grouped_fertilizers_t['Nutrient_nitrogen_N_(total)_t'] / grouped_fertilizers_t['Total_Fertilizers_t'],
    np.nan
)
grouped_fertilizers_t['Phosphate_Share_t'] = np.where(
    grouped_fertilizers_t['Total_Fertilizers_t'] > 0,
    grouped_fertilizers_t['Nutrient_phosphate_P2O5_(total)_t'] / grouped_fertilizers_t['Total_Fertilizers_t'],
    np.nan
)
grouped_fertilizers_t['Potash_Share_t'] = np.where(
    grouped_fertilizers_t['Total_Fertilizers_t'] > 0,
    grouped_fertilizers_t['Nutrient_potash_K2O_(total)_t'] / grouped_fertilizers_t['Total_Fertilizers_t'],
    np.nan
)

# Перевірка, які нові колонки додаються
columns_to_add_t = grouped_fertilizers_t.columns.difference(final_data.columns).to_list()

# Додавання нових колонок до оригінального DataFrame
final_data_t = pd.concat([final_data, grouped_fertilizers_t[columns_to_add_t]], axis=1)

# Групування даних за країною та роком
grouped_fertilizers_g = final_data_t.groupby(['Country', 'Year'], as_index=False).agg({
    'Nutrient_nitrogen_N_(total)_g/Int$': 'sum',
    'Nutrient_phosphate_P2O5_(total)_g/Int$': 'sum',
    'Nutrient_potash_K2O_(total)_g/Int$': 'sum'
})

# Обчислення загального обсягу добрив у г/Int$
grouped_fertilizers_g['Total_Fertilizers_g/Int$'] = (
    grouped_fertilizers_g['Nutrient_nitrogen_N_(total)_g/Int$'] +
    grouped_fertilizers_g['Nutrient_phosphate_P2O5_(total)_g/Int$'] +
    grouped_fertilizers_g['Nutrient_potash_K2O_(total)_g/Int$']
)

# Обчислення частки кожного добрива
grouped_fertilizers_g['Nitrogen_Share_g/Int$'] = np.where(
    grouped_fertilizers_g['Total_Fertilizers_g/Int$'] > 0,
    grouped_fertilizers_g['Nutrient_nitrogen_N_(total)_g/Int$'] / grouped_fertilizers_g['Total_Fertilizers_g/Int$'],
    np.nan
)
grouped_fertilizers_g['Phosphate_Share_g/Int$'] = np.where(
    grouped_fertilizers_g['Total_Fertilizers_g/Int$'] > 0,
    grouped_fertilizers_g['Nutrient_phosphate_P2O5_(total)_g/Int$'] / grouped_fertilizers_g['Total_Fertilizers_g/Int$'],
    np.nan
)
grouped_fertilizers_g['Potash_Share_g/Int$'] = np.where(
    grouped_fertilizers_g['Total_Fertilizers_g/Int$'] > 0,
    grouped_fertilizers_g['Nutrient_potash_K2O_(total)_g/Int$'] / grouped_fertilizers_g['Total_Fertilizers_g/Int$'],
    np.nan
)

# Перевірка, які нові колонки додаються
columns_to_add_g = grouped_fertilizers_g.columns.difference(final_data_t.columns).to_list()

# Додавання нових колонок до оригінального DataFrame
final_data_g = pd.concat([final_data_t, grouped_fertilizers_g[columns_to_add_g]], axis=1)

# Групування даних за країною та роком
grouped_pesticides = final_data_g.groupby(['Country', 'Year'], as_index=False).agg({
    'Pesticides_(total)_t': 'sum',
    'Pesticides_(total)_g/Int$': 'sum',
    'Agricultural_land_area_1000ha': 'sum'
})

# Обчислення кількості пестицидів на площу аграрної землі (тонни)
grouped_pesticides['Pesticides_per_agriculture_area_t'] = np.where(
    grouped_pesticides['Agricultural_land_area_1000ha'] > 0,
    grouped_pesticides['Pesticides_(total)_t'] / grouped_pesticides['Agricultural_land_area_1000ha'],
    np.nan
)

# Обчислення кількості пестицидів на площу аграрної землі (г/Int$)
grouped_pesticides['Pesticides_per_agriculture_area_g_per_Int'] = np.where(
    grouped_pesticides['Agricultural_land_area_1000ha'] > 0,
    grouped_pesticides['Pesticides_(total)_g/Int$'] / grouped_pesticides['Agricultural_land_area_1000ha'],
    np.nan
)

# Перевірка, які нові колонки додаються
columns_to_add = grouped_pesticides.columns.difference(final_data_g.columns).to_list()

# Додавання нових колонок до оригінального DataFrame
final_data = pd.concat([final_data_g, grouped_pesticides[columns_to_add]], axis=1)

# Видалення непотрібних колонок
columns_to_drop = [
    'Agricultural_land_area_1000ha',
    'Agriculture_area_1000ha',
    'Agriculture_area_actually_irrigated_area_1000ha',
    'Arable_land_area_1000ha',
    'Cropland_area_1000ha',
    'Cropland_area_actually_irrigated_area_1000ha',
    'Land_area_actually_irrigated_area_1000ha',
    'Land_area_area_1000ha',
    'Land_area_equipped_for_irrigation_area_1000ha',
    'Other_land_area_1000ha',
    'Perm._meadows_&_pastures_-_Cultivated_area_1000ha',
    'Perm._meadows_&_pastures_-_Nat._growing_area_1000ha',
    'Perm._meadows_&_pastures_area_actually_irrig._area_1000ha',
    'Permanent_crops_area_1000ha',
    'Permanent_meadows_and_pastures_area_1000ha',
    'Temporary_crops_area_1000ha',
    'Temporary_fallow_area_1000ha',
    'Temporary_meadows_and_pastures_area_1000ha',
    'Nutrient_nitrogen_N_(total)_t',
    'Nutrient_phosphate_P2O5_(total)_t',
    'Nutrient_potash_K2O_(total)_t',
    'Nutrient_nitrogen_N_(total)_g/Int$',
    'Nutrient_phosphate_P2O5_(total)_g/Int$',
    'Nutrient_potash_K2O_(total)_g/Int$',
    'Pesticides_(total)_t',
    'Pesticides_(total)_g/Int$',
    'Forest_land_area_1000_ha',
    'Forestry_area_actually_irrigated_area_1000_ha',
    'Naturally_regenerating_forest_area_1000_ha',
    'Planted_Forest_area_1000_ha',
    'Total_Fertilizers_g/Int$',
    'Total_Fertilizers_t'
]
final_data = final_data.drop(columns=columns_to_drop)

final_data.to_csv('datasets/final.csv', index=False)

# Вибираємо країну (наприклад, Німеччина)
country = 'Germany'
df_country = final_data[final_data['Country'] == country]

# Видалення колонок 'Country' та 'Year'
numeric_df = final_data.drop(columns=['Country', 'Year'])

plt.figure(figsize=(45, 45))  # Можна змінити розмір графіка відповідно до ваших даних
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Selected Columns)")

# Зберігаємо теплову карту в файл
plt.savefig('correlation_heatmap_world.png', dpi=300, bbox_inches='tight')  # Зберігаємо з високою якістю
plt.close()  # Закриваємо фігуру, щоб не відображалась на екрані
