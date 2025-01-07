import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Вимкнення warnings
warnings.filterwarnings("ignore")

# Завантаження датасету
df = pd.read_csv('datasets/aggregated_dataset_global.csv')

# Перетворення стовпця 'Year' на datetime
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

# Функція для підбору параметрів SARIMA
def optimize_sarima(ts, p_range, d_range, q_range, P_range, D_range, Q_range, S):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                model = SARIMAX(ts,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, S),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                model_fit = model.fit(disp=False)
                                if model_fit.aic < best_aic:
                                    best_aic = model_fit.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, S)
                                    best_model = model_fit
                            except Exception:
                                continue
    return best_order, best_seasonal_order, best_model

# Робота з кожним кластером
for cluster_num in df['Cluster'].unique():
    print(f"Обробка кластера {cluster_num}...")
    data = df[df['Cluster'] == cluster_num]
    data = data.sort_values(by='Date')  # Упорядковуємо за роками

    # Часовий ряд
    ts = data.set_index('Date')['Annual CO₂ emissions_global']

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

    # 2. Підбір параметрів SARIMA
    print("Підбір параметрів SARIMA...")
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    P_range = range(0, 3)
    D_range = range(0, 2)
    Q_range = range(0, 3)
    S = 12  # Річна сезонність (для місячних даних)

    best_order, best_seasonal_order, best_model = optimize_sarima(train, p_range, d_range, q_range, P_range, D_range, Q_range, S)
    print(f"Найкращі параметри для кластера {cluster_num}: {best_order} зі сезонними {best_seasonal_order}")

    # 3. Прогнозування
    print("Прогнозування...")
    y_pred = best_model.forecast(steps=len(test))

    # 4. Оцінка моделі
    mse = mean_squared_error(test, y_pred)
    mae = mean_absolute_error(test, y_pred)

    print(f"Оцінки для {cluster_num} кластера:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}\n")

    # 5. Візуалізація
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, label='Реальні значення')
    plt.plot(test.index, y_pred, label='Прогнозовані значення', linestyle='--')
    plt.xlabel('Рік')
    plt.ylabel('CO₂ Викиди')
    plt.title(f'SARIMA: Кластер {cluster_num}')
    plt.legend()
    plt.grid()
    plt.show()
