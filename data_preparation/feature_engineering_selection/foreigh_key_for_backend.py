import pandas as pd


df = pd.read_csv('datasets/filtered_emissions_data.csv')

# Створюємо унікальні списки країн та років
countries = df["Country"].unique()
years = df["Year"].unique()

# Створюємо словники для відповідності назва -> id
country_to_id = {country: idx + 1 for idx, country in enumerate(countries)}
year_to_id = {year: idx + 1 for idx, year in enumerate(years)}

# Додаємо колонки country_id і year_id
df["country_id"] = df["Country"].map(country_to_id)
df["year_id"] = df["Year"].map(year_to_id)

# Видаляємо оригінальні колонки Country і Year
df = df.drop(columns=["Country", "Year"])

# Зберігаємо новий CSV
df.to_csv("dataset_with_indices.csv", index=False)

# Зберігаємо словники для перевірки (опціонально)
pd.DataFrame(list(country_to_id.items()), columns=["Country", "ID"]).to_csv("countries_indices.csv", index=False)
pd.DataFrame(list(year_to_id.items()), columns=["Year", "ID"]).to_csv("years_indices.csv", index=False)

print("Новий файл збережено як 'dataset_with_indices.csv'")
