import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


df = pd.read_csv('../feature_engineering_selection/datasets/final.csv')

# Вибір колонок, які потрібні
columns_of_interest = [
    'Country',
    'Year',
    'Annual CO₂ emissions',
    'Fossil Share Consumption',
    'Non-CO₂ Share Consumption',
    'Nuclear Share Consumption',
    'Renewable Share Consumption',
    'Fossil Electricity Share',
    'Non-CO₂ Electricity Share',
    'Nuclear Electricity Share',
    'Renewable Electricity Share'
]

df_selected = df[columns_of_interest]

# Видаляємо рядки з пропусками по економічних показниках
economic_columns = [
    'Fossil Electricity Share',
    'Non-CO₂ Electricity Share',
    'Nuclear Electricity Share',
    'Renewable Electricity Share'
]

df_selected = df_selected[df_selected[economic_columns].notnull().any(axis=1)]

# Видаляємо рядки з пропусками у викидах CO2
df_selected = df_selected[df_selected['Annual CO₂ emissions'].notnull()]

# Вибір даних за останній рік (наприклад, 2020)
df_2020 = df_selected[df_selected['Year'] == 2020]

# Масштабування даних для кластеризації
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_2020[economic_columns])

# Метод "ліктя" для вибору оптимальної кількості кластерів
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Візуалізація методу "ліктя"
plt.figure(figsize=(10, 8))
plt.plot(k_range, inertia, marker='o')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів в групі енергетичних факторів')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Інерція')
plt.grid(True)
plt.show()
