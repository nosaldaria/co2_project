import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Завантаження датасету
df = pd.read_csv('datasets/aggregated_dataset_global_new.csv')

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

# Словник з найкращими параметрами для кожного кластера
best_params = {
    0: {'n_estimators': 200},
    1: {'n_estimators': 200},
    2: {'n_estimators': 50},
    3: {'n_estimators': 50},
}

# Масив для збереження прогнозів
predictions = []

# Обробка кожного кластера
for cluster_num, features in cluster_features.items():
    # Завантаження даних для поточного кластера
    data = df[df['Cluster'] == cluster_num]  # Фільтруємо за кластером

    # Перевірка на пропущені значення та їх обробка
    data = data.dropna(subset=features + ['Annual CO₂ emissions'])  # Видаляємо рядки з пропусками

    X = data[features]  # Використовуємо відповідні фактори для кожного кластера
    y = data['Annual CO₂ emissions']  # Залежна змінна

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Моделювання з Random Forest
    params = best_params[cluster_num]  # Отримуємо параметри з словника
    rf = RandomForestRegressor(random_state=42, **params)  # Передаємо параметри в модель
    rf.fit(X_train, y_train)

    # Прогнозування на тестовому наборі
    y_pred = rf.predict(X_test)

    # Додаємо прогноз до масиву
    predictions.append(y_pred)

    # Оцінка моделі
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Виведемо результати
    print(f'Результати для {cluster_num} кластеру:')
    print(f'Середня квадратична помилка (MSE): {mse}')
    print(f'Коефіцієнт детермінації (R²): {r2}')
    print(f'Середня абсолютна помилка (MAE): {mae}')
    print(f'Корінь середньої квадратичної помилки (RMSE): {rmse}')
    print(f'Середня абсолютна відносна помилка (MAPE): {mape}%\n')

    # Візуалізація результатів
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title(f'Прогнозування для {cluster_num} кластеру за допомогою Random Forest')
    plt.xlabel('Реальні значення CO₂ emissions')
    plt.ylabel('Прогнозовані значення CO₂ emissions')
    plt.show()

    # Налаштування параметрів для пошуку
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],  # Кількість дерев
        'max_depth': [None, 10, 20, 30, 50],  # Максимальна глибина дерев
        'min_samples_split': [2, 5, 10, 15],  # Мінімальна кількість зразків для поділу вузла
        'min_samples_leaf': [1, 2, 4, 8],  # Мінімальна кількість зразків у листі
        'max_features': ['auto', 'sqrt', 'log2']  # Кількість особливостей, що використовуються на кожному вузлі
    }

    # Ініціалізація моделі Random Forest
    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='r2',  # Метрика оцінки
        cv=5,  # Кількість розбиттів для крос-валідації
        verbose=2,
        n_jobs=-1  # Використовувати всі ядра процесора
    )

    grid_search.fit(X_train, y_train)

    print(f"Найкращі параметри: {grid_search.best_params_}")

# Об'єднання прогнозів (середнє по всіх кластерах)
global_predictions = np.mean(predictions, axis=0)

# Виведемо результат
plt.figure(figsize=(8, 6))
plt.scatter(y_test, global_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Об\'єднані прогнози для всіх кластерів за допомогою Random Forest')
plt.xlabel('Реальні значення CO₂ emissions')
plt.ylabel('Об\'єднані прогнози CO₂ emissions')
plt.show()
