from django.shortcuts import render
from django.http import JsonResponse
from .models import Countries, Year, Emissions, ForecastResults


# Головна сторінка (мапа)
def map_view(request):
    data = Emissions.objects.select_related('country', 'year')  # Отримуємо всі дані
    countries_data = data.values('country__country_name', 'annual_co2_emissions')
    context = {'emissions': countries_data}
    return render(request, 'map.html', context)


# Таблиця
def table_view(request):
    data = Emissions.objects.select_related('country', 'year')  # Отримуємо дані для таблиці
    context = {'emissions': data}
    return render(request, 'table.html', context)


def chart_view(request):
    emissions_data = Emissions.objects.values('year__Year', 'country__country_name', 'annual_co2_emissions')
    chart_data = list(emissions_data)
    context = {'chart_data': chart_data}
    return render(request, 'charts.html', context)


# Інформація про проект
def info_view(request):
    project_info = {
        'title': 'CO2 Emissions Forecasting',
        'description': 'This project provides data visualization, analysis, and forecasting of CO2 emissions by country.',
        'methods': ['Ridge Regression', 'ARIMA', 'Random Forest', 'XGBoost'],
        'objectives': 'To provide insights into global CO2 emissions and their drivers.',
        'data_sources': 'The data is sourced from global emissions datasets and projected based on historical trends.'
    }
    return render(request, 'info.html', {'info': project_info})


# API для отримання даних у форматі JSON (опціонально)
def emissions_api(request):
    country_name = request.GET.get('country', None)
    year = request.GET.get('year', None)
    if country_name and year:
        data = Emissions.objects.filter(country__country_name=country_name, year__Year=year).values(
            'country__country_name', 'year__Year', 'annual_co2_emissions')
    else:
        data = list(Emissions.objects.select_related('country', 'year').values(
            'country__country_name', 'year__Year', 'annual_co2_emissions'))
    return JsonResponse(data, safe=False)

