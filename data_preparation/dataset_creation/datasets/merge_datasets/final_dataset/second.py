import pandas as pd
import numpy as np
from functools import reduce

import seaborn as sns
import matplotlib.pyplot as plt


f_df = pd.read_csv('first_dataset.csv')
s_df = pd.read_csv('../agricultur/datasets/agriculture_features.csv')
t_df = pd.read_csv('../forest/datasets/forest_data.csv')

print(f_df.head())  # Перевіряємо загальну структуру
print(s_df.head())  # Перевіряємо загальну структуру
print(t_df.head())  # Перевіряємо загальну структуру

merged_data = reduce(
    lambda left, right: pd.merge(left, right, on=["Country", "Year"], how="outer", validate="one_to_one"),
    [f_df, s_df, t_df]
)

print(merged_data.info())  # Перевіряємо загальну структуру
print(merged_data.head())  # Дивимося на перші рядки

merged_data.to_csv('second_dataset.csv', index=False)

# Видалення колонок 'Country' та 'Year'
numeric_df = merged_data.drop(columns=['Country', 'Year'])

plt.figure(figsize=(25, 25))  # Задайте ширину та висоту графіка
# Теплова карта кореляцій
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
# Зберігаємо теплову карту в файл
plt.savefig('correlation_heatmap_second.png', dpi=300, bbox_inches='tight')  # Зберігаємо з високою якістю
plt.close()  # Закриваємо фігуру, щоб не відображалась на екрані
