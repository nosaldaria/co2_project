import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Завантажуємо дані
df = pd.read_csv('../../feature_engineering_selection/final.csv')

# Визначаємо колонки для аналізу
columns_of_interest = [
    'Country', 'Year', 'Annual CO₂ emissions',
    'Agricultural_landshare_area_%',
    'Croplandshare_area_%',
    'Permanent_meadows_and_pasturesshare_area_%',
    'Agriculture_area_actually_irrigatedshare_agriculture_%',
    'Arable_landshare_agriculture_%',
    'Croplandshare_agriculture_%',
    'Permanent_cropsshare_agriculture_%',
    'Permanent_meadows_and_pasturesshare_agriculture_%',
    'Other_land_share',
    'Temporary_crops_area_agriculture_share',
    'Nitrogen_Share_t',
    'Phosphate_Share_t',
    'Potash_Share_t',
    'Nitrogen_Share_g/Int$',
    'Phosphate_Share_g/Int$',
    'Potash_Share_g/Int$',
    'Pesticides_per_agriculture_area_g_per_Int',
    'Pesticides_per_agriculture_area_t',
    'Forest_land_share_land_area_%',
    'Naturally_regenerating_forest_share_forest_land_%',
]

# Очищаємо дані: фільтруємо по економічних та соціальних факторах без пропусків
economic_columns = columns_of_interest[3:]  # Оскільки перші три — не економічні

df_selected = df[columns_of_interest].dropna(subset=economic_columns)

# Список колонок для перевірки пропусків та інтерполяції
columns_to_interpolate = [
    'Agricultural_landshare_area_%',
    'Croplandshare_area_%',
    'Arable_landshare_agriculture_%',
    'Potash_Share_g/Int$',
    'Pesticides_per_agriculture_area_g_per_Int',
    'Forest_land_share_land_area_%',
    'Naturally_regenerating_forest_share_forest_land_%',
]

# Перевірка пропусків для кожної країни по всіх колонках
valid_countries = df_selected.groupby('Country').apply(
    lambda x: x[columns_to_interpolate].notnull().any(axis=1).any()
)

# Видаляємо рядки з пропусками у викидах CO2
df_selected = df_selected[df_selected['Annual CO₂ emissions'].notnull()]

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
    'Agricultural_landshare_area_%',
    'Croplandshare_area_%',
    'Arable_landshare_agriculture_%',
    'Potash_Share_g/Int$',
    'Pesticides_per_agriculture_area_g_per_Int',
    'Forest_land_share_land_area_%',
    'Naturally_regenerating_forest_share_forest_land_%',
]

# Логарифмічна трансформація для зменшення впливу аномалій
for column in columns_to_transform:
    df_2020.loc[:, f'Log {column.split("_")[0]}'] = np.log(df_2020[column] + 1)

# Масштабуємо дані
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_2020[['Log Agricultural',
             'Log Croplandshare',
             'Log Arable',
             'Log Potash',
             'Log Pesticides',
             'Log Forest',
             'Log Naturally']])

# Кластеризація за допомогою KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_2020['Cluster'] = kmeans.fit_predict(X_scaled)

# Виведення результатів
print(df_2020[['Country',
               'Agricultural_landshare_area_%',
               'Croplandshare_area_%',
               'Arable_landshare_agriculture_%',
               'Potash_Share_g/Int$',
               'Pesticides_per_agriculture_area_g_per_Int',
               'Forest_land_share_land_area_%',
               'Naturally_regenerating_forest_share_forest_land_%',
               'Cluster']])

# Візуалізація
plt.figure(figsize=(15, 10))
plt.scatter(df_2020['Country'], df_2020['Agricultural_landshare_area_%'], c=df_2020['Cluster'], cmap='viridis')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Agricultural_landshare_area_%')
plt.title('Clustering Countries by Agricultural landshare area (2020)')
plt.show()

df_2020.to_csv('agriculture_clusters.csv', index=False)

# Видаляємо колонку 'Year' з df_2020, щоб уникнути конфліктів
df_2020_clusters = df_2020[['Country', 'Cluster']]

# Отримуємо список країн із кластерів
selected_countries = df_2020_clusters['Country'].unique()

# Фільтруємо початковий датасет для цих країн
filtered_data = df[df['Country'].isin(selected_countries)].copy()

filtered_data = filtered_data[columns_of_interest]

# Додаємо колонку з кластерами, зіставляючи її з df_2020_clusters
filtered_data = filtered_data.merge(df_2020_clusters, on='Country', how='left')

# Перевіряємо результат
print(filtered_data.head())

# Зберігаємо новий датасет
filtered_data.to_csv('datasets/dataset_with_agriculture_clusters.csv', index=False)

# # Агрегуємо середні значення економіко-соціальних факторів і суми викидів CO₂
aggregated_data = filtered_data.groupby(['Year', 'Cluster']).agg({
    'Annual CO₂ emissions': 'sum',
    'Agricultural_landshare_area_%': 'mean',
    'Croplandshare_area_%': 'mean',
    'Permanent_meadows_and_pasturesshare_area_%': 'mean',
    'Agriculture_area_actually_irrigatedshare_agriculture_%': 'mean',
    'Arable_landshare_agriculture_%': 'mean',
    'Croplandshare_agriculture_%': 'mean',
    'Permanent_cropsshare_agriculture_%': 'mean',
    'Permanent_meadows_and_pasturesshare_agriculture_%': 'mean',
    'Other_land_share': 'mean',
    'Temporary_crops_area_agriculture_share': 'mean',
    'Nitrogen_Share_t': 'mean',
    'Phosphate_Share_t': 'mean',
    'Potash_Share_t': 'mean',
    'Nitrogen_Share_g/Int$': 'mean',
    'Phosphate_Share_g/Int$': 'mean',
    'Potash_Share_g/Int$': 'mean',
    'Pesticides_per_agriculture_area_g_per_Int': 'mean',
    'Pesticides_per_agriculture_area_t': 'mean',
    'Forest_land_share_land_area_%': 'mean',
    'Naturally_regenerating_forest_share_forest_land_%': 'mean',
}).reset_index()

# Збереження результату в CSV
aggregated_data.to_csv('aggregated_by_cluster_and_year_agriculture.csv', index=False)

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
