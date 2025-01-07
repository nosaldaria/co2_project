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

# Функція для підбору параметрів SARIMAX
def optimize_sarimax(ts, exog, p_range, d_range, q_range, seasonal_order):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = SARIMAX(ts, exog=exog, order=(p, d, q), seasonal_order=seasonal_order)
                    model_fit = model.fit(disp=False)
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception:
                    continue
    return best_order, best_model

cluster_features = {
    0: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    1: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    2: ['Imports of goods and services (% of GDP)', 'Exports of goods and services (% of GDP)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)'],
    3: ['Exports per capita (current US$)', 'Imports per capita (current US$)',
        'GDP per capita (current US$)', 'Adjusted savings: carbon dioxide damage (% of total population)',
        'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows per capita (BoP, current US$)',
        'Population density (people per sq. km of land area)', 'Urban population (% of total population)']
}

# Робота з кожним кластером
for cluster_num in df['Cluster'].unique():
    print(f"Обробка кластера {cluster_num}...")
    data = df[df['Cluster'] == cluster_num]
    data = data.sort_values(by='Date')  # Упорядковуємо за роками

    # Часовий ряд
    ts = data.set_index('Date')['Annual CO₂ emissions_global']

    # Вибір екзогенних змінних (незалежних змінних) згідно з кластером
    exog_columns = cluster_features.get(cluster_num, ['GDP', 'Electricity Consumption', 'Population'])  # За замовчуванням
    exog = data[exog_columns]

    # Розділення на тренувальну та тестову вибірки
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    exog_train, exog_test = exog[:train_size], exog[train_size:]

    # 1. Перевірка стаціонарності
    print("Перевірка стаціонарності...")
    check_stationarity(train)

    # Якщо ряд не стаціонарний, робимо диференціювання
    ts_diff = train.diff().dropna()

    print("Перевірка стаціонарності після диференціювання...")
    check_stationarity(ts_diff)

    # 2. Підбір параметрів SARIMAX
    print("Підбір параметрів SARIMAX...")
    p_range = range(0, 12)  # Зазначаємо діапазон для p
    d_range = range(0, 12)  # Зазначаємо діапазон для d
    q_range = range(0, 12)  # Зазначаємо діапазон для q
    seasonal_order = (1, 1, 1, 12)  # Наприклад, сезонність з періодом 12 (для місяців)
    # Підбір параметрів SARIMAX
    best_order, best_model = optimize_sarimax(train, exog_train, p_range, d_range, q_range, seasonal_order)

    # Перевірка наявності моделі перед прогнозуванням
    if best_model is None:
        print(f"Не вдалося знайти кращі параметри для кластера {cluster_num}. Пропускаємо цей кластер.")
        continue  # Перехід до наступного кластера
    else:
        print(f"Найкращі параметри для кластера {cluster_num}: {best_order}")

    # 3. Прогнозування
    print("Прогнозування...")
    y_pred = best_model.forecast(steps=len(test), exog=exog_test)

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
    plt.title(f'SARIMAX: Кластер {cluster_num}')
    plt.legend()
    plt.grid()
    plt.show()
