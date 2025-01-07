import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
df = pd.read_csv('datasets/aggregated_dataset_global.csv')

# Розділення даних за кластерами
clusters = df['Cluster'].unique()
results = {}


def prepare_data(data, features, target, timesteps=12):
    """Підготовка даних для LSTM."""
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    data_scaled = feature_scaler.fit_transform(data[features])
    target_scaled = target_scaler.fit_transform(data[[target]])

    X, y = [], []
    for i in range(timesteps, len(data_scaled)):
        X.append(data_scaled[i - timesteps:i])
        y.append(target_scaled[i])

    return np.array(X), np.array(y), feature_scaler, target_scaler


# Цільова змінна для прогнозування
target = 'Annual CO₂ emissions_global'

for cluster in clusters:
    print(f"Моделюємо для кластера {cluster}...")

    # Дані для кластера
    cluster_data = df[df['Cluster'] == cluster]

    # Особливості для поточного кластера
    features = cluster_features[cluster]

    # Підготовка даних
    X, y, feature_scaler, target_scaler = prepare_data(cluster_data, features, target)

    # Розділення на тренувальний і тестовий набори
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Побудова моделі LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Навчання
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)

    # Прогнозування
    y_pred = model.predict(X_test)

    # Відновлення масштабу цільової змінної
    y_test_rescaled = target_scaler.inverse_transform(y_test)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred)

    # Оцінка
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

    results[cluster] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# Виведення результатів
for cluster, metrics in results.items():
    print(f"Кластер {cluster}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
