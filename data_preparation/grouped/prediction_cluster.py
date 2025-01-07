from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('datasets/new_dataset.csv')

# Додаємо нові ознаки
additional_columns = [
    'Exports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Imports of goods and services (% of GDP)',
    'Population density (people per sq. km of land area)',
    'Urban population (% of total population)',
    'Adjusted savings: carbon dioxide damage (% of GDP)'
]

# Групуємо дані по кластерах і роках
aggregated_data = df.groupby(['Cluster', 'Year']).agg({
    'Annual CO₂ emissions': 'sum',
    'GDP per capita (current US$)': 'mean',
    'Exports of goods and services (% of GDP)': 'mean',
    'Final consumption expenditure (% of GDP)': 'mean',
    'Foreign direct investment, net inflows (% of GDP)': 'mean',
    'Imports of goods and services (% of GDP)': 'mean',
    'Population density (people per sq. km of land area)': 'mean',
    'Urban population (% of total population)': 'mean',
    'Adjusted savings: carbon dioxide damage (% of GDP)': 'mean'
}).reset_index()

# Видаляємо рядки з пропущеними значеннями
aggregated_data = aggregated_data.dropna()

# Розділяємо на X (фактори) та y (цільова змінна)
X = aggregated_data[[
    'Cluster', 'Year', 'GDP per capita (current US$)',
    'Exports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Imports of goods and services (% of GDP)',
    'Population density (people per sq. km of land area)',
    'Urban population (% of total population)',
    'Adjusted savings: carbon dioxide damage (% of GDP)'
]]
y = aggregated_data['Annual CO₂ emissions']

# Кодуємо кластер як числовий фактор
X.loc[:, 'Cluster'] = X['Cluster'].astype('category').cat.codes

# Стандартизуємо ознаки та цільову змінну
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Створюємо словник для зберігання моделей для кожного кластера
models = {}

# Створюємо окремі моделі для кожного кластера
for cluster in sorted(X['Cluster'].unique()):
    print(f"\nTraining model for Cluster {cluster}...")

    # Фільтруємо дані для поточного кластера
    X_cluster = X_scaled[X['Cluster'] == cluster]
    y_cluster = y_scaled[X['Cluster'] == cluster]

    # Розділяємо на навчальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)

    # Навчаємо лінійну регресію
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Зберігаємо модель в словник
    models[cluster] = model

    # Прогнозуємо
    y_pred = model.predict(X_test)

    # Оцінюємо модель
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Cluster {cluster}: Mean Squared Error: {mse}, R² Score: {r2}")

    # Візуалізуємо результати
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.xlabel('Actual CO₂ Emissions')
    plt.ylabel('Predicted CO₂ Emissions')
    plt.title(f'Cluster {cluster}: Actual vs Predicted CO₂ Emissions')
    plt.grid(True)
    plt.show()

    # Прогноз на наступний рік для поточного кластера
    next_year = max(aggregated_data['Year'][aggregated_data['Cluster'] == cluster]) + 1
    avg_gdp = aggregated_data[aggregated_data['Cluster'] == cluster]['GDP per capita (current US$)'].mean()
    avg_exports = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Exports of goods and services (% of GDP)'].mean()
    avg_consumption = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Final consumption expenditure (% of GDP)'].mean()
    avg_fdi = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Foreign direct investment, net inflows (% of GDP)'].mean()
    avg_imports = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Imports of goods and services (% of GDP)'].mean()
    avg_density = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Population density (people per sq. km of land area)'].mean()
    avg_urban_pop = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Urban population (% of total population)'].mean()
    avg_savings = aggregated_data[aggregated_data['Cluster'] == cluster][
        'Adjusted savings: carbon dioxide damage (% of GDP)'].mean()

    # Прогноз для наступного року
    prediction_scaled = model.predict(
        scaler_X.transform([[cluster, next_year, avg_gdp, avg_exports, avg_consumption, avg_fdi,
                             avg_imports, avg_density, avg_urban_pop, avg_savings]]))
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

    print(f"Cluster {cluster}, Year {next_year}, Predicted CO₂ Emissions: {prediction[0][0]:.2f}")
