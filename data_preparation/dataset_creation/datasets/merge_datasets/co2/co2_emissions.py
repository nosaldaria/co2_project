import pandas as pd


co2_per_capita = pd.read_csv('../../datasets/co2/co-emissions-per-capita.csv')
co2_per_country = pd.read_csv('../../datasets/co2/annual-co-emissions.csv')
co2_cumulative = pd.read_csv('../../datasets/co2/cumulative-co-emissions.csv')
co2_per_dollar_gdp = pd.read_csv('../../datasets/co2/carbon-intensity-co-emissions-per-dollar-of-gdp.csv')

co2 = [
    co2_per_capita,
    co2_per_country,
    co2_cumulative,
    co2_per_dollar_gdp
]

for co2_metric in co2:
    print(f'{co2_metric.head()}')

merged_data = co2_per_capita.merge(
    co2_per_country, on=['Entity', 'Year'], how='outer', validate='one_to_one'
).merge(
    co2_cumulative, on=['Entity', 'Year'], how='outer', validate='one_to_one'
).merge(
    co2_per_dollar_gdp, on=['Entity', 'Year'], how='outer', validate='one_to_one'
)

merged_data = merged_data[merged_data['Year'] >= 1990]
merged_data = merged_data.drop(columns=['Code'])
merged_data.rename(columns={merged_data.columns[0]: 'Country'}, inplace=True)

merged_data.to_csv("datasets/co2_emissions_combined.csv", index=False)

print(merged_data.head())

# Об'єднаємо дані про викиди з даними про споживання енергії та економічними показниками
economic = pd.read_csv('../economic/datasets/merged_dataset_new.csv')

merged_data = merged_data.merge(
    economic,
    on=['Country', 'Year'],  # Спільні колонки
    how='outer',  # Тільки спільні країни та роки
    validate='one_to_one'
)

print(merged_data.head())

merged_data.to_csv('datasets/merged_data_new.csv', index=False)
