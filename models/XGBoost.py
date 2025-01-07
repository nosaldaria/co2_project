import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Словник для збереження списків змінних для кожного кластера
cluster_features = {
    0: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    1: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)',
        'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    2: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)',
        'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    3: ['Exports per capita (current US$)', 'Imports per capita (current US$)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)',
        'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)']
}

# Завантаження даних
df = pd.read_csv('datasets/aggregated_dataset_global_new.csv')

# Масив для збереження прогнозів
predictions = []

# Словник для збереження результатів
xgboost_results = {}

# Цільова змінна для прогнозування
target = 'Annual CO₂ emissions'

# Розділення даних за кластерами
clusters = df['Cluster'].unique()

for cluster in clusters:
    print(f"Моделюємо для кластера {cluster} за допомогою XGBoost...")

    # Дані для поточного кластера
    cluster_data = df[df['Cluster'] == cluster]

    # Особливості для поточного кластера
    features = cluster_features[cluster]

    # Підготовка даних
    X = cluster_data[features].values
    y = cluster_data[target].values

    # Розділення на тренувальний і тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабування даних
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ініціалізація та тренування XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Прогнозування
    y_pred = model.predict(X_test_scaled)

    # Додаємо прогноз до масиву
    predictions.append(y_pred)

    # Оцінка
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Збереження результатів
    xgboost_results[cluster] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'y_test': y_test,
        'y_pred': y_pred
    }

    # Візуалізація результатів
    plt.figure(figsize=(12, 6))

    # Реальні vs Прогнозовані значення
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.title(f'Cluster {cluster}: Real vs Predicted CO2 Emissions')
    plt.xlabel('Real CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')

    # Помилки прогнозування
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=20, color='green', alpha=0.7)
    plt.title(f'Cluster {cluster}: Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Важливість ознак
    plt.figure(figsize=(20, 10))
    feature_importances = model.feature_importances_
    plt.barh(features, feature_importances, color='orange')
    plt.title(f'Cluster {cluster}: Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Об'єднання прогнозів для всіх кластерів
global_predictions = np.mean(predictions, axis=0)

# Візуалізація об'єднаних прогнозів
plt.figure(figsize=(8, 6))
plt.scatter(y_test, global_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Об\'єднані прогнози для всіх кластерів за допомогою XGBoost')
plt.xlabel('Реальні значення CO₂ emissions')
plt.ylabel('Об\'єднані прогнози CO₂ emissions')
plt.show()

# Виведення результатів для кожного кластера
for cluster, metrics in xgboost_results.items():
    print(f"Кластер {cluster}:")
    for metric, value in metrics.items():
        if metric not in ['y_test', 'y_pred']:  # Не виводити прогнози
            print(f"  {metric}: {value}")
