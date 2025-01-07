import pandas as pd


def clean_and_filter_data(xl, sheet_name, output_filename, start=3, end=102, del_col=3, start_year=1990):
    """
    Функція для обробки та фільтрації даних з Excel:
    - Завантажує дані з зазначеного листа.
    - Видаляє порожні рядки.
    - Перетворює дані в long формат.
    - Фільтрує дані за роками (за замовчуванням з 1990 року).
    - Зберігає очищені дані у CSV файл.

    Parameters:
    - xl: ExcelFile об'єкт, що містить всі дані.
    - sheet_name: назва листа, з якого потрібно завантажити дані.
    - output_filename: ім'я для збереженого файлу CSV.
    - start_year: рік, з якого потрібно почати (за замовчуванням 1990).

    Returns:
    - df_filtered: очищений та відфільтрований DataFrame.
    """

    # Завантаження даних з листа
    df = xl.parse(sheet_name, header=None)

    # Видалення порожніх рядків
    df.dropna(how='all', inplace=True)

    # Перший рядок стає заголовками колонок
    df.columns = df.iloc[2]
    df = df[start:end]  # Видаляємо зайві рядки (пусті чи метадані)

    # Видаляємо останні 3 колонки
    df = df.iloc[:, :-del_col]

    # Перейменовуємо колонку країни в "Country"
    df.rename(columns={df.columns[0]: 'Country'}, inplace=True)

    # Видалення рядків з порожнім значенням у першій колонці
    df = df[df['Country'].notna()]

    # Перетворення даних у long формат
    df_long = df.melt(id_vars=['Country'], var_name='Year', value_name='Energy Consumption')

    # Перетворення року на числовий формат
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')

    # Перетворення року на ціле число
    df_long['Year'] = df_long['Year'].astype(int)

    # Видалення рядків з некоректними роками
    df_long = df_long.dropna(subset=['Year']).reset_index(drop=True)

    # Фільтруємо дані, залишаючи лише роки >= start_year
    df_filtered = df_long[df_long['Year'] >= start_year]

    df_filtered.to_csv(output_filename, index=False)

    return df_filtered


def combine_energy_data(dfs):
    """
    Об'єднує різні датафрейми споживання енергії в один.
    Кожен датафрейм має колонку 'Country' та 'Energy Consumption', а також 'Energy Source' для кожного виду енергії.
    Після об'єднання, кожен вид енергії буде в окремій колонці.
    """

    # Додаємо колонки з ім'ям джерела енергії для кожного датафрейму
    for df, name in dfs:
        df['Energy Source'] = name

    # Об'єднуємо всі датафрейми в один
    combined_df = pd.concat([df for df, _ in dfs], ignore_index=True)

    # Тепер для кожної країни створюємо окрему колонку для кожного виду енергії
    pivoted_df = combined_df.pivot_table(index=['Country', 'Year'], columns='Energy Source',
                                         values='Energy Consumption', aggfunc='sum')

    # Скидаємо рівні для колонок
    pivoted_df.reset_index(inplace=True)

    return pivoted_df


xl = pd.ExcelFile("../../EI-Stats-Review-All-Data.xlsx")

# Завантаження даних із листа "Primary energy cons - EJ"
df_primary_energy = clean_and_filter_data(xl, 'Primary energy cons - EJ',
                                          "../../datasets/renewable_energy/Primary_energy.csv")

oil_production_tonnes = clean_and_filter_data(xl, 'Oil Production - tonnes',
                                              "../../datasets/fossil_fuels/production/Oil_production_tonnes.csv",
                                              3, 67)
oil_consumption_ej = clean_and_filter_data(xl, 'Oil Consumption - EJ',
                                           "../../datasets/fossil_fuels/consumption/Oil_consumption_ej.csv")
oil_input_electricity_generation = clean_and_filter_data(xl, 'Oil inputs - Elec generation ',
                                                         "../../datasets/fossil_fuels/input_electricity/Oil_inputs_electricity_generation_ej.csv",
                                                         3, 47)

gas_production_ej = clean_and_filter_data(xl, 'Gas Production - EJ',
                                          "../../datasets/fossil_fuels/production/Gas_production_ej.csv",
                                          3, 66)
gas_consumption_ej = clean_and_filter_data(xl, 'Gas Consumption - EJ',
                                           "../../datasets/fossil_fuels/consumption/Gas_consumption_ej.csv")
gas_input_electricity_generation = clean_and_filter_data(xl, 'Gas inputs - Elec generation',
                                                         "../../datasets/fossil_fuels/input_electricity/Gas_inputs_electricity_generation_ej.csv",
                                                         3, 47)

coal_production_ej = clean_and_filter_data(xl, 'Coal Production - EJ',
                                           "../../datasets/fossil_fuels/production/Coal_production_ej.csv",
                                           3, 50)
coal_consumption_ej = clean_and_filter_data(xl, 'Coal Consumption - EJ',
                                            "../../datasets/fossil_fuels/consumption/Coal_consumption_ej.csv")
coal_input_electricity_generation = clean_and_filter_data(xl, 'Coal inputs - Elec generation ',
                                                          "../../datasets/fossil_fuels/input_electricity/Coal_inputs_electricity_generation_ej.csv",
                                                          3, 48)

nuclear_generation = clean_and_filter_data(xl, 'Nuclear Generation - TWh',
                                           "../../datasets/renewable_energy/generation/nuclear_generation.csv",
                                           3, 108)
nuclear_consumption = clean_and_filter_data(xl, 'Nuclear Consumption - EJ',
                                            "../../datasets/renewable_energy/consumption/nuclear_consumption_ej.csv",
                                            3, 108)

hydro_generation = clean_and_filter_data(xl, 'Hydro Generation - TWh',
                                         "../../datasets/renewable_energy/generation/hydro_generation.csv")
hydro_consumption = clean_and_filter_data(xl, 'Hydro Consumption - EJ',
                                          "../../datasets/renewable_energy/consumption/hydro_consumption_ej.csv")

solar_generation = clean_and_filter_data(xl, 'Solar Generation - TWh',
                                         "../../datasets/renewable_energy/generation/solar_generation.csv")
solar_consumption = clean_and_filter_data(xl, 'Solar Consumption - EJ',
                                          "../../datasets/renewable_energy/consumption/solar_consumption_ej.csv")

wind_generation = clean_and_filter_data(xl, 'Wind Generation - TWh',
                                        "../../datasets/renewable_energy/generation/wind_generation.csv")
wind_consumption = clean_and_filter_data(xl, 'Wind Consumption - EJ',
                                         "../../datasets/renewable_energy/consumption/wind_consumption_ej.csv")


dfs = [
    (df_primary_energy, 'Primary Energy Consumption'),
    (oil_consumption_ej, 'Oil Consumption'),
    (gas_consumption_ej, 'Gas Consumption'),
    (coal_consumption_ej, 'Coal Consumption'),
    (nuclear_consumption, 'Nuclear Consumption'),
    (hydro_consumption, 'Hydro Consumption'),
    (solar_consumption, 'Solar Consumption'),
    (wind_consumption, 'Wind Consumption')
]

# Об'єднання даних
combined_data = combine_energy_data(dfs)

combined_data.to_csv("datasets/combined_energy_consumption.csv", index=False)
