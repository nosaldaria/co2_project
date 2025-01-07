import pandas as pd
import numpy as np


df = pd.read_csv('datasets/dataset.csv')
countries = pd.read_csv('datasets/world-data-2023.csv')

# Отримуємо унікальні країни з датасету
unique_countries_main = df['Country'].drop_duplicates()

# Перевіряємо, чи всі унікальні країни з вашого датасету є в списку країн
missing_countries = unique_countries_main[~unique_countries_main.isin(countries['Country'])]

# Виводимо країни, яких не вистачає в списку країн
print(missing_countries)

# Зберігаємо пропущені країни в новий датасет
missing_countries_df = pd.DataFrame(missing_countries)

missing_countries_df.to_csv('datasets/missing_countries.csv', index=False)

territories_to_remove = [
    'Antarctica', 'Aruba', 'Belgium-Luxembourg', 'Bermuda', 'Bonaire, Sint Eustatius and Saba',
    'Bonaire Sint Eustatius and Saba', 'British Virgin Islands', 'Cayman Islands',
    'Channel Islands', 'Christmas Island', 'Curacao', 'Curaçao', 'Czechoslovakia', 'Falkland Islands (Malvinas)',
    'Faroe Islands', 'French Guiana', 'French Polynesia', 'Gibraltar', 'Guadeloupe', 'Guam', 'International aviation',
    'International shipping', 'Ireland', 'Isle of Man', 'Kosovo', 'Kuwaiti Oil Fires (GCP)', 'Macao', 'Macao SAR, China',
    'Martinique', 'Mayotte', 'Montserrat', 'New Caledonia', 'Niue', 'Norfolk Island', 'Pitcairn', 'Puerto Rico',
    'Réunion', 'Ryukyu Islands (GCP)', 'Saint Barthélemy', 'Saint Helena', 'Saint Helena, Ascension and Tristan da Cunha',
    'Saint Martin (French part)', 'Saint Pierre and Miquelon', 'Serbia and Montenegro', 'Sint Maarten (Dutch part)',
    'St. Martin (French part)', 'Sudan (former)', 'Tokelau', 'Turks and Caicos Islands', 'USSR', 'United States Virgin Islands',
    'Virgin Islands (U.S.)', 'Wallis and Futuna', 'Wallis and Futuna Islands', 'Yugoslav SFR', 'Anguilla', 'Cook Islands',
    'Greenland', 'Taiwan', 'Netherlands Antilles (former)', 'Pacific Islands Trust Territory', 'Sao Tome and Principe',
    'China, Taiwan Province of', 'Taiwan', 'China Hong Kong SAR', 'China, Hong Kong SAR', 'China, Macao SAR', 'Hong Kong SAR, China',
    'Macao SAR, China', 'Hong Kong', 'Macao', 'West Bank and Gaza', 'China, mainland', 'Holy See', 'Western Sahara',
    'China, mainland', 'San Marino', 'Monaco', 'Northern Mariana Islands'
]

# Видалення цих територій
df_main_cleaned = df[~df['Country'].isin(territories_to_remove)]

df_main_cleaned.to_csv('datasets/cleaned_main_dataset.csv', index=False)

# Створюємо мапу різних назв для об'єднання країн
name_mapping = {
    'American Samoa': 'Samoa',
    'Samoa': 'Samoa',
    'Bahamas': 'The Bahamas',
    'Bahamas, The': 'The Bahamas',
    'Congo': 'Republic of the Congo',
    'Congo, Rep.': 'Republic of the Congo',
    'Republic of Congo ': 'Republic of the Congo',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Democratic Republic of Congo': 'Democratic Republic of the Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
    'Bolivia': 'Bolivia',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Venezuela': 'Venezuela',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Venezuela, RB': 'Venezuela',
    'Brunei': 'Brunei',
    'Brunei Darussalam': 'Brunei',
    'Cabo Verde': 'Cape Verde',
    'Cape Verde': 'Cape Verde',
    'Czechia': 'Czech Republic',
    'Czech Republic': 'Czech Republic',
    "Cote d'Ivoire": 'Ivory Coast',
    "Côte d'Ivoire": 'Ivory Coast',
    "Democratic People's Republic of Korea": 'North Korea',
    "Korea, Dem. People's Rep.": 'North Korea',
    'North Korea': 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Republic of Korea': 'South Korea',
    'South Korea': 'South Korea',
    'Egypt, Arab Rep.': 'Egypt',
    'Egypt': 'Egypt',
    'Ethiopia': 'Ethiopia',
    'Ethiopia PDR': 'Ethiopia',
    'Gambia': 'The Gambia',
    'Gambia, The': 'The Gambia',
    'Iran': 'Iran',
    'Iran (Islamic Republic of)': 'Iran',
    'Iran, Islamic Rep.': 'Iran',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Kyrgyzstan': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    "Lao People's Democratic Republic": 'Laos',
    'Laos': 'Laos',
    'Micronesia (Federated States of)': 'Federated States of Micronesia',
    'Micronesia (country)': 'Federated States of Micronesia',
    'Micronesia, Fed. Sts.': 'Federated States of Micronesia',
    'Netherlands': 'Netherlands',
    'Netherlands (Kingdom of the)': 'Netherlands',
    'Republic of Moldova': 'Moldova',
    'Moldova': 'Moldova',
    'Russia': 'Russia',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Slovakia': 'Slovakia',
    'Syria': 'Syria',
    'Syrian Arab Republic': 'Syria',
    'East Timor': 'East Timor',
    'Timor-Leste': 'East Timor',
    'Trinidad & Tobago': 'Trinidad and Tobago',
    'Trinidad and Tobago': 'Trinidad and Tobago',
    'Turkey': 'Turkey',
    'Turkiye': 'Turkey',
    'Türkiye': 'Turkey',
    'US': 'United States',
    'United States': 'United States',
    'United States of America': 'United States',
    'Ireland': 'Republic of Ireland',
    'United Kingdom': 'United Kingdom',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Tanzania': 'Tanzania',
    'United Republic of Tanzania': 'Tanzania',
    'Viet Nam': 'Vietnam',
    'Vietnam': 'Vietnam',
    'Yemen': 'Yemen',
    'Yemen, Rep.': 'Yemen',
    'Saint Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'Saint Lucia': 'Saint Lucia',
    'St. Lucia': 'Saint Lucia',
    'Saint Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Palestine': 'Palestinian National Authority',
}

# Змінюємо назви країн згідно з мапою
df_main_cleaned['Country'] = df_main_cleaned['Country'].replace(name_mapping)

# Функція для агрегування
def custom_agg(x):
    if x.isnull().all():
        return np.nan
    else:
        return x.sum(min_count=1)  # Використовуємо суму тільки для числових значень

# Застосування custom_agg
df_aggregated = (
    df_main_cleaned
    .groupby(['Country', 'Year'])
    .agg(custom_agg)
    .reset_index()
)

df_aggregated.to_csv('final_dataset.csv', index=False)

