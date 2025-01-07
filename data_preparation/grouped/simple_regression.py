import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('datasets/dataset.csv')

# Припустимо, що в колонці 'Income Group' є категорії (наприклад, 'Low Income', 'High Income')
X = df[['GDP per capita (current US$)', 'Income Group']]
y = df['CO2 Emissions']

# Розділяємо на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створюємо pipeline для обробки категоріальних та числових змінних
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['GDP per capita (current US$)', 'Energy Consumption', 'Agriculture Share']),
        ('cat', OneHotEncoder(), ['Income Group'])
    ])

# Створюємо модель з RandomForest, який враховує категоріальні змінні
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Навчаємо модель
model.fit(X_train, y_train)

# Прогнозуємо
y_pred = model.predict(X_test)
