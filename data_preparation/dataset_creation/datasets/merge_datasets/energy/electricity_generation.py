import pandas as pd


xl = pd.ExcelFile("../../EI-Stats-Review-All-Data.xlsx")

# Завантаження даних з листа
df = xl.parse("Electricity Generation - TWh", header=None)

# Видалення порожніх рядків
df = df.dropna(how='all')

# Перший рядок стає заголовками колонок
df.columns = df.iloc[2]
df = df[3:101]  # Видаляємо зайві рядки (пусті чи метадані)

# Видаляємо останні 3 колонки
df = df.iloc[:, :-3]

# Перейменовуємо колонку країни в "Country"
df.rename(columns={df.columns[0]: 'Country'}, inplace=True)

# Видалення рядків з порожнім значенням у першій колонці
df = df[df['Country'].notna()]

# Перетворення даних у long формат
df_long = df.melt(id_vars=['Country'], var_name='Year', value_name='Electricity Generation Twh')

# Перетворення року на числовий формат
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')

# Перетворення року на ціле число
df_long['Year'] = df_long['Year'].astype(int)

# Видалення рядків з некоректними роками
df_long = df_long.dropna(subset=['Year']).reset_index(drop=True)

# Фільтруємо дані, залишаючи лише роки >= start_year
df_filtered = df_long[df_long['Year'] >= 1990]

df_filtered.to_csv("datasets/electricity_generation.csv", index=False)
