import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Завантажуємо дані
df = pd.read_csv('../../feature_engineering_selection/datasets/final.csv')

# Визначаємо колонки для аналізу
columns_of_interest = [
    'Country', 'Year', 'Annual CO₂ emissions',
    'Exports of goods and services (% of GDP)', 'Final consumption expenditure (% of GDP)',
    'Foreign direct investment, net inflows (% of GDP)', 'GDP per capita (current US$)',
    'Imports of goods and services (% of GDP)', 'Population density (people per sq. km of land area)',
    'Urban population (% of total population)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
    'Adjusted savings: carbon dioxide damage (% of total population)',
    'Exports per capita (current US$)', 'Imports per capita (current US$)',
    'Foreign direct investment, net inflows per capita (BoP, current US$)'
]

# Очищаємо дані: фільтруємо по економічних та соціальних факторах без пропусків
economic_columns = columns_of_interest[3:]  # Оскільки перші три — не економічні

df_selected = df[columns_of_interest].dropna(subset=economic_columns)

# Список колонок для перевірки пропусків та інтерполяції
columns_to_interpolate = [
    'GDP per capita (current US$)'
]

# Перевірка пропусків для кожної країни по всіх колонках
valid_countries = df_selected.groupby('Country').apply(
    lambda x: x[columns_to_interpolate].notnull().any(axis=1).any()
)

# Фільтруємо країни без пропусків
df_selected_cleaned = df_selected[df_selected['Country'].isin(valid_countries[valid_countries].index)]

# Лінійна інтерполяція для колонок з пропусками
for column in columns_to_interpolate:
    df_selected_cleaned[column] = df_selected_cleaned.groupby('Country')[column].transform(
        lambda x: x.interpolate(method='linear'))

# Перевірка пропусків після інтерполяції
missing_after_interpolation = df_selected_cleaned[columns_to_interpolate].isnull().sum()
print("Пропуски після інтерполяції:")
print(missing_after_interpolation)

# Видалення залишкових пропусків
df_selected_cleaned = df_selected_cleaned.dropna(subset=columns_to_interpolate)

# Перевірка пропусків після очищення
missing_after_cleaning = df_selected_cleaned[columns_to_interpolate].isnull().sum()
print("Пропуски після очищення:")
print(missing_after_cleaning)

# Завантажуємо та фільтруємо дані за 2020 рік
df_2020 = df_selected_cleaned[df_selected_cleaned['Year'] == 2020]

# Створюємо список колонок для логарифмічної трансформації
columns_to_transform = [
    'GDP per capita (current US$)'
]

# Логарифмічна трансформація для зменшення впливу аномалій
for column in columns_to_transform:
    df_2020.loc[:, f'Log {column.split(" ")[0]}'] = np.log(df_2020[column] + 1)

# Масштабуємо дані
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_2020[['Log GDP']])

# Кластеризація за допомогою KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df_2020['Cluster'] = kmeans.fit_predict(X_scaled)

# Виведення результатів
print(df_2020[['Country', 'GDP per capita (current US$)',
               'Cluster']])

# Візуалізація
plt.figure(figsize=(15, 10))
plt.scatter(df_2020['Country'], df_2020['GDP per capita (current US$)'], c=df_2020['Cluster'], cmap='viridis')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('GDP per capita (current US$)')
plt.title('Clustering Countries by GDP per capita (2020)')
plt.show()

# Видаляємо колонку 'Year' з df_2020, щоб уникнути конфліктів
df_2020_clusters = df_2020[['Country', 'Cluster']]

# Отримуємо список країн із кластерів
selected_countries = df_2020_clusters['Country'].unique()

# Фільтруємо початковий датасет для цих країн
filtered_data = df[df['Country'].isin(selected_countries)].copy()

filtered_data = filtered_data[columns_of_interest]

# Додаємо колонку з кластерами, зіставляючи її з df_2020_clusters
filtered_data = filtered_data.merge(df_2020_clusters, on='Country', how='left')

# Зберігаємо новий датасет з кластерами
filtered_data.to_csv('datasets/dataset_with_economics_clusters.csv', index=False)

# Агрегуємо середні значення економіко-соціальних факторів і суми викидів CO₂
aggregated_data = filtered_data.groupby(['Year', 'Cluster']).agg({
    'Annual CO₂ emissions': 'sum',  # Сума викидів CO₂
    'Exports of goods and services (% of GDP)': 'mean',
    'Final consumption expenditure (% of GDP)': 'mean',
    'Foreign direct investment, net inflows (% of GDP)': 'mean',
    'GDP per capita (current US$)': 'mean',
    'Imports of goods and services (% of GDP)': 'mean',
    'Population density (people per sq. km of land area)': 'mean',
    'Urban population (% of total population)': 'mean',
    'Adjusted savings: carbon dioxide damage (% of GDP)': 'mean',
    'Adjusted savings: carbon dioxide damage (% of total population)': 'mean',
    'Exports per capita (current US$)': 'mean',
    'Imports per capita (current US$)': 'mean',
    'Foreign direct investment, net inflows per capita (BoP, current US$)': 'mean'
}).reset_index()

# Збереження результату в CSV
aggregated_data.to_csv('aggregated_by_cluster_and_year.csv', index=False)

# Перевіряємо результати
print(aggregated_data.head())

# Створюємо окремі мапи кореляцій для кожного кластера
clusters = aggregated_data['Cluster'].unique()

for cluster in clusters:
    cluster_data = aggregated_data[aggregated_data['Cluster'] == cluster]

    # Видаляємо нечислові або некорисні колонки
    numeric_data = cluster_data.drop(columns=['Year', 'Cluster'])

    # Обчислюємо кореляційну матрицю
    correlation_matrix = numeric_data.corr()

    # Візуалізуємо хітмапу
    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'Correlation Map for Cluster {cluster}')
    plt.show()
