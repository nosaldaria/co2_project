# emissions/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/emissions/', views.EmissionsList.as_view(), name='emissions-list'),
    path('api/forecast/', views.ForecastResultsList.as_view(), name='forecast-list'),
    path('api/emissions/<int:country_id>/', views.EmissionsByCountry.as_view(), name='emissions-by-country'),
]
