from django.urls import path
from . import views  # Імпортуємо в'юшки з вашого додатка

urlpatterns = [
    path('', views.table_view, name='home'),         # Головна сторінка -> мапа
    path('table/', views.table_view, name='table'), # Таблиця
    path('map/', views.map_view, name='map'),      # Мапа
    path('charts/', views.chart_view, name='charts'), # Графіки
    path('info/', views.info_view, name='info'),   # Інформація про проект
]
