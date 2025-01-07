import pandas as pd

# Завантаження обох датасетів
emissions_data = pd.read_csv('datasets/final_dataset.csv')  # Датасет з викидами
countries_data = pd.read_csv('datasets/world-data-2023.csv')  # Датасет з країнами

# Фільтрація першого датасету за наявністю країн у другому
filtered_emissions_data = emissions_data[emissions_data['Country'].isin(countries_data['Country'])]

# Створення датасету з країнами, яких немає в першому датасеті
missing_countries = countries_data[~countries_data['Country'].isin(emissions_data['Country'])]

filtered_emissions_data = filtered_emissions_data[filtered_emissions_data['Year'] == 1990]

# Збереження результатів у нові CSV файли
filtered_emissions_data.to_csv('filtered_emissions_data_1990.csv', index=False)
missing_countries.to_csv('missing_countries_new.csv', index=False)

print("Фільтрація завершена. Датасети збережено.")
