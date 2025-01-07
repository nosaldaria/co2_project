import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('datasets/co2_drops.csv')

df['Adjusted savings: carbon dioxide damage (% of GDP)'] = df['Adjusted savings: carbon dioxide damage (current US$)'] / df['GDP (current US$)'] * 100
df['Adjusted savings: carbon dioxide damage (% of total population)'] = df['Adjusted savings: carbon dioxide damage (current US$)'] / df['Population, total']
df['Exports per capita (current US$)'] = df['Exports of goods and services (current US$)'] / df['Population, total']
df['Imports per capita (current US$)'] = df['Imports of goods and services (current US$)'] / df['Population, total']
df['Foreign direct investment, net inflows per capita (BoP, current US$)'] = df['Foreign direct investment, net inflows (BoP, current US$)'] / df['Population, total']

columns_to_drop = [
    'Adjusted savings: carbon dioxide damage (current US$)', 'Exports of goods and services (BoP, current US$)',
    'Exports of goods and services (annual % growth)', 'Exports of goods and services (current US$)',
    'Final consumption expenditure (annual % growth)', 'Final consumption expenditure (current US$)',
    'Foreign direct investment, net inflows (BoP, current US$)', 'GDP (current US$)', 'GDP growth (annual %)',
    'GDP per capita growth (annual %)', 'Imports of goods and services (BoP, current US$)',
    'Imports of goods and services (annual % growth)', 'Imports of goods and services (current US$)',
    'Population growth (annual %)', 'Population, total', 'Urban population', 'Urban population growth (annual %)']
df = df.drop(columns=columns_to_drop)

# Вибираємо країну (наприклад, Німеччина)
country = 'Ukraine'
df_country = df[df['Country'] == country]

numeric_df = df_country.drop(columns=['Country', 'Year'])

# Вибір колонок для аналізу
columns_of_interest = ['Annual CO₂ emissions', 'GDP per capita (current US$)',
                       'Exports of goods and services (% of GDP)',
                       'Final consumption expenditure (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)',
                       'Imports of goods and services (% of GDP)', 'Population density (people per sq. km of land area)',
                       'Urban population (% of total population)', 'Adjusted savings: carbon dioxide damage (% of GDP)',
                       'Adjusted savings: carbon dioxide damage (% of total population)', 'Imports per capita (current US$)',
                       'Exports per capita (current US$)', 'Foreign direct investment, net inflows per capita (BoP, current US$)'
                       ]

# Створення підмножини DataFrame з обраними колонками
subset_df = numeric_df[columns_of_interest]

# Теплова карта кореляцій
plt.figure(figsize=(45, 45))  # Можна змінити розмір графіка відповідно до ваших даних
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Selected Columns)")

# Зберігаємо теплову карту в файл
plt.savefig('correlation_heatmap_ukraine.png', dpi=300, bbox_inches='tight')  # Зберігаємо з високою якістю
plt.close()  # Закриваємо фігуру, щоб не відображалась на екрані

df.to_csv('datasets/dataset_with_new_economics_features.csv', index=False)
