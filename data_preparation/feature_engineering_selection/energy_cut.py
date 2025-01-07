import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('datasets/dataset_with_new_economics_features.csv')

print(df.info())

# Групування за країною та роком
grouped_electricity = df.groupby(['Country', 'Year'], as_index=False).agg({
    'Nuclear_TWh': 'sum',
    'Hydro_TWh': 'sum',
    'Solar_TWh': 'sum',
    'Wind_TWh': 'sum',
    'Oil_Electricity': 'sum',
    'Gas_Electricity': 'sum',
    'Coal_Electricity': 'sum',
    'Electricity Generation Twh': 'sum'
})

# Обчислення частки відновлювальних джерел
grouped_electricity['Renewable Electricity Share'] = np.where(
    grouped_electricity['Electricity Generation Twh'] > 0,
    (grouped_electricity['Hydro_TWh'] +
     grouped_electricity['Solar_TWh'] +
     grouped_electricity['Wind_TWh']) / grouped_electricity['Electricity Generation Twh'],
    np.nan
)

# Обчислення частки атомної енергії
grouped_electricity['Nuclear Electricity Share'] = np.where(
    grouped_electricity['Electricity Generation Twh'] > 0,
    grouped_electricity['Nuclear_TWh'] / grouped_electricity['Electricity Generation Twh'],
    np.nan
)

# Обчислення частки викопних джерел
grouped_electricity['Fossil Electricity Share'] = np.where(
    grouped_electricity['Electricity Generation Twh'] > 0,
    (grouped_electricity['Oil_Electricity'] +
     grouped_electricity['Gas_Electricity'] +
     grouped_electricity['Coal_Electricity']) / grouped_electricity['Electricity Generation Twh'],
    np.nan
)

# Обчислення частки генерації без CO₂
grouped_electricity['Non-CO₂ Electricity Share'] = np.where(
    grouped_electricity['Electricity Generation Twh'] > 0,
    (grouped_electricity['Nuclear_TWh'] +
     grouped_electricity['Hydro_TWh'] +
     grouped_electricity['Solar_TWh'] +
     grouped_electricity['Wind_TWh']) / grouped_electricity['Electricity Generation Twh'],
    np.nan
)

# Забезпечуємо, що дубльовані колонки не додаються:
# Вибираємо тільки ті колонки, які відсутні в оригінальному датасеті
columns_to_add = grouped_electricity.columns.difference(df.columns).to_list()
final_data = pd.concat([df, grouped_electricity[columns_to_add]], axis=1)

# Групування за країною та роком
grouped_production = final_data.groupby(['Country', 'Year'], as_index=False).agg({
    'Oil_Production': 'sum',
    'Gas_Production': 'sum',
    'Coal_Production': 'sum',
})

# Обчислення Total Fossil Production, ігноруючи NaN
grouped_production['Total Fossil Production'] = grouped_production[['Oil_Production', 'Gas_Production', 'Coal_Production']].sum(axis=1, skipna=True)

# Якщо всі значення NaN, залишити Total Fossil Production як NaN
grouped_production.loc[
    grouped_production[['Oil_Production', 'Gas_Production', 'Coal_Production']].isna().all(axis=1),
    'Total Fossil Production'
] = np.nan

# Обчислення часток видобутку для кожного типу викопного палива
grouped_production['Oil Production Share'] = grouped_production['Oil_Production'] / grouped_production['Total Fossil Production']
grouped_production['Gas Production Share'] = grouped_production['Gas_Production'] / grouped_production['Total Fossil Production']
grouped_production['Coal Production Share'] = grouped_production['Coal_Production'] / grouped_production['Total Fossil Production']

columns_to_add = grouped_production.columns.difference(final_data.columns).to_list()
final_data = pd.concat([final_data, grouped_production[columns_to_add]], axis=1)

# Видалення непотрібних колонок
columns_to_drop = [
    'Nuclear_TWh',
    'Hydro_TWh',
    'Solar_TWh',
    'Wind_TWh',
    'Oil_Electricity',
    'Gas_Electricity',
    'Coal_Electricity',
    'Coal_Production',
    'Oil_Production',
    'Gas_Production',
    'Total Fossil Production',
    'Primary Energy Consumption',
    'Electricity Generation Twh'
]
final_data = final_data.drop(columns=columns_to_drop)

final_data.to_csv('datasets/electricity_generation_share.csv', index=False)

# Вибираємо країну (наприклад, Німеччина)
country = 'Ukraine'
df_country = final_data[final_data['Country'] == country]

# Видалення колонок 'Country' та 'Year'
numeric_df = df_country.drop(columns=['Country', 'Year'])

# Вибір колонок для аналізу
# columns_of_interest = ['Annual CO₂ emissions', 'GDP per capita (current US$)',
#                        'Exports of goods and services (% of GDP)',
#                        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)',
#                        'Imports of goods and services (% of GDP)', 'Population density (people per sq. km of land area)',
#                        'Urban population (% of total population)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
#                        'Adjusted savings: carbon dioxide damage (% of total population)', 'Imports per capita (current US$)',
#                        'Exports per capita (current US$)', 'Foreign direct investment, net inflows per capita (BoP, current US$)'
#                        ]
#
# # Створення підмножини DataFrame з обраними колонками
# subset_df = numeric_df[columns_of_interest]

# Теплова карта кореляцій

plt.figure(figsize=(45, 45))  # Можна змінити розмір графіка відповідно до ваших даних
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Selected Columns)")

# Зберігаємо теплову карту в файл
plt.savefig('correlation_heatmap_ukraine_energy_gen_share.png', dpi=300, bbox_inches='tight')  # Зберігаємо з високою якістю
plt.close()  # Закриваємо фігуру, щоб не відображалась на екрані

# final_data.to_csv('dataset_with_new_economics_features.csv', index=False)
