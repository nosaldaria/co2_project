import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Вимкнення warnings
warnings.filterwarnings("ignore")

# Завантаження датасету
df = pd.read_csv('datasets/aggregated_dataset_global_new.csv')

# Перетворення стовпця 'Date' на datetime
df['Date'] = pd.to_datetime(df['Date'])

# Функція для перевірки стаціонарності
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    if result[1] <= 0.05:
        print("Часовий ряд стаціонарний.\n")
    else:
        print("Часовий ряд не стаціонарний.\n")

# Функція для підбору параметрів ARIMA
def optimize_arima(ts, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception:
                    continue
    return best_order, best_model

# Масив для збереження прогнозів
predictions = []

# Робота з кожним кластером
for cluster_num in df['Cluster'].unique():
    print(f"Обробка кластера {cluster_num}...")
    data = df[df['Cluster'] == cluster_num]
    data = data.sort_values(by='Date')  # Упорядковуємо за роками

    # Часовий ряд
    ts = data.set_index('Date')['Annual CO₂ emissions']

    # Розділення на тренувальну та тестову вибірки
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    # 1. Перевірка стаціонарності
    print("Перевірка стаціонарності...")
    check_stationarity(train)

    # Якщо ряд не стаціонарний, робимо диференціювання
    ts_diff = train.diff().dropna()

    print("Перевірка стаціонарності після диференціювання...")
    check_stationarity(ts_diff)

    # 2. Підбір параметрів ARIMA
    print("Підбір параметрів ARIMA...")
    p_range = range(0, 3)  # Зазначаємо діапазон для p
    d_range = range(0, 3)  # Зазначаємо діапазон для d
    q_range = range(0, 3)  # Зазначаємо діапазон для q
    best_order, best_model = optimize_arima(train, p_range, d_range, q_range)
    print(f"Найкращі параметри для кластера {cluster_num}: {best_order}")

    # 3. Прогнозування
    print("Прогнозування...")
    y_pred = best_model.forecast(steps=len(test))

    # 4. Оцінка моделі
    mse = mean_squared_error(test, y_pred)
    mae = mean_absolute_error(test, y_pred)
    r2 = r2_score(test, y_pred)

    print(f"Оцінки для {cluster_num} кластера:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}\n")
    print(f'Коефіцієнт детермінації (R²): {r2}')

    # Додаємо прогноз до масиву
    predictions.append(y_pred)

    # 5. Візуалізація
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, label='Реальні значення')
    plt.plot(test.index, y_pred, label='Прогнозовані значення', linestyle='--')
    plt.xlabel('Рік')
    plt.ylabel('CO₂ Викиди')
    plt.title(f'ARIMA: Кластер {cluster_num}')
    plt.legend()
    plt.grid()
    plt.show()

# Об'єднання прогнозів (середнє по всіх кластерах)
global_predictions = np.mean(predictions, axis=0)

# Виведемо результат
plt.figure(figsize=(8, 6))
plt.plot(test.index, np.mean(predictions, axis=0), label="Об'єднані прогнози", linestyle='--', color='orange')
plt.plot(test.index, test, label="Реальні значення", color='blue')
plt.xlabel('Дата')
plt.ylabel('CO₂ Викиди')
plt.title('Об\'єднані прогнози для всіх кластерів за допомогою ARIMA')
plt.legend()
plt.grid()
plt.show()
