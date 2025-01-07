import pandas as pd
import numpy as np
from functools import reduce

import seaborn as sns
import matplotlib.pyplot as plt


# first dataset
f_df = pd.read_csv('../co2/datasets/merged_data_new.csv')
s_df = pd.read_csv('../energy/datasets/electricity_generation.csv')
t_df = pd.read_csv('../energy/datasets/renewable_energy_combined.csv')
fo_df = pd.read_csv('../fossil_fuel/datasets/fossil_fuels_combined.csv')
fi_df = pd.read_csv('../fossil_fuel/datasets/fossil_electricity_combined.csv')

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [f_df, s_df, t_df, fo_df, fi_df]
)

# merged_data.fillna(0, inplace=True)
# Замінити ".." на NaN
merged_data.replace("..", np.nan, inplace=True)

print(merged_data.info())  # Перевіряємо загальну структуру
print(merged_data.head())  # Дивимося на перші рядки

merged_data.to_csv('first_dataset.csv', index=False)

# Видалення колонок 'Country' та 'Year'
# numeric_df = merged_data.drop(columns=['Country', 'Year'])
#
# plt.figure(figsize=(25, 25))  # Задайте ширину та висоту графіка
# # Теплова карта кореляцій
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap")
# # Зберігаємо теплову карту в файл
# plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')  # Зберігаємо з високою якістю
# plt.close()  # Закриваємо фігуру, щоб не відображалась на екрані
