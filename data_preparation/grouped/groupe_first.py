import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('../feature_engineering_selection/datasets/final.csv')

# Визначаємо список колонок, які потрібно вибрати
columns_of_interest = [
    'Country',
    'Year',
    'Annual CO₂ emissions',
    'Exports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Foreign direct investment, net inflows (% of GDP)',
    'GDP per capita (current US$)',
    'Imports of goods and services (% of GDP)',
    'Population density (people per sq. km of land area)',
    'Urban population (% of total population)',
    'Adjusted savings: carbon dioxide damage (% of GDP)',
    'Adjusted savings: carbon dioxide damage (% of total population)',
    'Exports per capita (current US$)',
    'Imports per capita (current US$)',
    'Foreign direct investment, net inflows per capita (BoP, current US$)'
]

# Створюємо новий DataFrame з лише потрібними колонками
df_selected = df[columns_of_interest]

# Окремо визначаємо колонки з економіко-соціальними факторами
economic_columns = [
    'Exports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Foreign direct investment, net inflows (% of GDP)',
    'GDP per capita (current US$)',
    'Imports of goods and services (% of GDP)',
    'Population density (people per sq. km of land area)',
    'Urban population (% of total population)',
    'Adjusted savings: carbon dioxide damage (% of GDP)',
    'Adjusted savings: carbon dioxide damage (% of total population)',
    'Exports per capita (current US$)',
    'Imports per capita (current US$)',
    'Foreign direct investment, net inflows per capita (BoP, current US$)'
]

# Перевіряємо пропуски тільки по економіко-соціальних факторах
df_selected = df_selected[df_selected[economic_columns].notnull().any(axis=1)]

df_selected.to_csv('datasets/data.csv', index=False)

# Перевіряємо, чи є хоча б одне значення по ВВП на душу населення для кожної країни за всі роки
valid_countries = df_selected.groupby('Country')['GDP per capita (current US$)'].apply(lambda x: x.notnull().any())

# Залишаємо тільки країни, де є хоча б одне значення по ВВП
df_selected_cleaned = df_selected[df_selected['Country'].isin(valid_countries[valid_countries].index)]

df_selected_cleaned['GDP per capita (current US$)'] = df_selected_cleaned.groupby('Country')['GDP per capita (current US$)'].transform(lambda x: x.interpolate(method='linear'))

# Перевіряємо кількість пропусків після інтерполяції
missing_after_interpolation = df_selected_cleaned['GDP per capita (current US$)'].isnull().sum()
print(f"Пропуски після лінійної інтерполяції: {missing_after_interpolation}")

df_selected_cleaned = df_selected_cleaned.dropna(subset=['GDP per capita (current US$)'])

print("Пропуски після інтерполяції:", df_selected_cleaned['GDP per capita (current US$)'].isnull().sum())

df_selected_cleaned.to_csv('datasets/cleaned_data.csv', index=False)

# Фільтруємо дані за 2020 рік
df_2020 = df_selected_cleaned[df_selected_cleaned['Year'] == 2020]

# Логарифмічна трансформація для зменшення впливу аномалій
df_2020['Log GDP'] = np.log(df_2020['GDP per capita (current US$)'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_2020[['Log GDP']])

kmeans = KMeans(n_clusters=4, random_state=42)
df_2020['Cluster'] = kmeans.fit_predict(X_scaled)

print(df_2020[['Country', 'GDP per capita (current US$)', 'Cluster']])

# Візуалізуємо результати
plt.figure(figsize=(15,12))
plt.scatter(df_2020['Country'], df_2020['GDP per capita (current US$)'], c=df_2020['Cluster'], cmap='viridis')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('GDP per capita (current US$)')
plt.title('Clustering Countries by GDP per capita (2020)')
plt.show()

df_2020.to_csv('datasets/cluster_data.csv', index=False)

data = df_selected[df_selected['Country'] != 'North Korea']

# Видаляємо колонку Year з df_2020, щоб уникнути конфліктів
df_2020_clusters = df_2020[['Country', 'Cluster']]

# Додаємо колонку з кластерами до основного датасету
data = data.merge(df_2020_clusters, on='Country', how='left')

print(data[['Country', 'Cluster']].head())

data.to_csv('datasets/new_dataset.csv', index=False)

# Метод "ліктя" для вибору оптимальної кількості кластерів
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Візуалізація методу "ліктя"
plt.figure(figsize=(12, 10))
plt.plot(k_range, inertia, marker='o')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів в групі економіко-соціальних факторів')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Інерція')
plt.grid(True)
plt.show()
