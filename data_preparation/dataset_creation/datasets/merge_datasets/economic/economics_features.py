import pandas as pd
import numpy as np


df = pd.read_csv("../../datasets/economics/economics_features.csv")

# Видалення рядків, починаючи з 5201
df = df.iloc[:5208]  # Обрізаємо до 5200-го рядка

# Перетворення датасету: з колонок-років в одну колонку
df_melted = df.melt(
    id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],  # Колонки, які залишаються незмінними
    var_name="Year",  # Нова колонка для років
    value_name="Value"  # Нова колонка для значень
)

# Видалення непотрібних колонок
df_melted = df_melted.drop(columns=["Country Code", "Series Code"], errors="ignore")

# Видалення зайвих символів "[YRxxxx]" з років
df_melted["Year"] = df_melted["Year"].str.extract(r'(\d{4})')

# Використовуємо pivot для перетворення Series Name в окремі колонки
df_pivoted = df_melted.pivot_table(
    index=["Country Name", "Year"],  # Залишаємо Country Name та Year як індекси
    columns="Series Name",  # Створюємо колонки для кожного унікального значення в Series Name
    values="Value",  # Заповнюємо нові колонки значеннями з Value
    aggfunc="first"  # Якщо є кілька значень для однієї пари (Country Name, Year), беремо перше
)

df_pivoted = df_pivoted.reset_index()

df_pivoted.to_csv("economics_features_new.csv", index=False)
