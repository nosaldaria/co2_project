import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Завантаження датасету
df = pd.read_csv('datasets/aggregated_dataset_global_new.csv')

# Перетворення стовпця Date на тип datetime
df['Date'] = pd.to_datetime(df['Date'])

# Словник для збереження списків змінних для кожного кластера
cluster_features = {
    0: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],  # Зміни для 0 кластера
    1: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],  # Зміни для 1 кластера
    2: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],  # Зміни для 2 кластера
    3: ['Exports per capita (current US$)', 'Imports per capita (current US$)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)']  # Зміни для 3 кластера
}

# Словник для значень alpha для кожного кластера
alpha_values = {
    0: 0.2,
    1: 0.2,
    2: 0.2,
    3: 0.2
}

# Масив для збереження прогнозів
predictions = []

# Обробка кожного кластера
for cluster_num, features in cluster_features.items():
    # Завантаження даних для поточного кластера
    data = df[df['Cluster'] == cluster_num]
    X = data[features]
    y = data['Annual CO₂ emissions']

    # Стандартизація
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Моделювання з Ridge регресією
    ridge = Ridge(alpha=0.2)
    ridge.fit(X_train, y_train)

    # Прогнозування на тестовому наборі
    y_pred = ridge.predict(X_test)

    # Додаємо прогноз до масиву
    predictions.append(y_pred)

    # Оцінка моделі
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Виведемо результати для поточного кластера
    print(f'Результати для {cluster_num} кластеру:')
    print(f'Середня квадратична помилка (MSE): {mse}')
    print(f'Коефіцієнт детермінації (R²): {r2}')
    print(f'Середня абсолютна помилка (MAE): {mae}')
    print(f'Корінь середньої квадратичної помилки (RMSE): {rmse}')
    print(f'Середня абсолютна відносна помилка (MAPE): {mape}%\n')

    # 6. Візуалізація результатів (на приклад тестових та передбачених значень)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title(f'Прогнозування для {cluster_num} кластеру за допомогою Ridge регресії')
    plt.xlabel('Реальні значення CO₂ emissions')
    plt.ylabel('Прогнозовані значення CO₂ emissions')
    plt.show()

    # Шукали найкращий коефіцієнт для моделі
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    ridge = Ridge()

    alpha_values = np.concatenate([
        np.linspace(0.1, 1, 100),       # 100 значень від 0.1 до 1
        np.arange(1, 10, 100),          # 100 значень від 1 до 10
        np.arange(10, 100, 100)         # 100 значень від 10 до 100
    ])

    # Округлення
    alpha_values = np.round(alpha_values, 3)

    # Сітка значень для alpha
    param_grid = {'alpha': alpha_values}

    # Пошук найкращого значення alpha
    grid_search = GridSearchCV(ridge, param_grid, cv=5)  # 5-fold cross-validation
    grid_search.fit(X_train, y_train)

    # Виведення найкращого значення alpha
    print(f"Best alpha: {grid_search.best_params_['alpha']}")

# Об'єднання прогнозів (середнє по всіх кластерах)
global_predictions = np.mean(predictions, axis=0)

# Виведемо результат
plt.figure(figsize=(8, 6))
plt.scatter(y_test, global_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Об\'єднані прогнози для всіх кластерів')
plt.xlabel('Реальні значення CO₂ emissions')
plt.ylabel('Об\'єднані прогнози CO₂ emissions')
plt.show()

