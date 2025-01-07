import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('../feature_engineering_selection/datasets/final.csv')

# Вибір потрібних колонок
columns_of_interest = [
    'Country',
    'Year',
    'Annual CO₂ emissions',
    'Agricultural_landshare_area_%',
    'Croplandshare_area_%',
    'Arable_landshare_agriculture_%',
    'Potash_Share_g/Int$',
    'Pesticides_per_agriculture_area_g_per_Int',
]

df_selected = df[columns_of_interest]

# Видаляємо рядки з пропусками у вибраних колонках
agricultural_columns = [
    'Agricultural_landshare_area_%',
    'Croplandshare_area_%',
    'Arable_landshare_agriculture_%',
    'Potash_Share_g/Int$',
    'Pesticides_per_agriculture_area_g_per_Int'
]

df_selected = df_selected[df_selected[agricultural_columns].notnull().all(axis=1)]

# Видаляємо рядки з пропусками у викидах CO2
df_selected = df_selected[df_selected['Annual CO₂ emissions'].notnull()]

# Вибір даних за останній рік (наприклад, 2020)
df_2020 = df_selected[df_selected['Year'] == 2020]

# Масштабування даних для кластеризації
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_2020[agricultural_columns])

# Метод "ліктя" для вибору оптимальної кількості кластерів
inertia = []
k_range = range(1, 11)  # Від 1 до 10 кластерів

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Візуалізація методу "ліктя"
plt.figure(figsize=(10, 8))
plt.plot(k_range, inertia, marker='o')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів в групі с/г факторів')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Інерція')
plt.grid(True)
plt.show()
