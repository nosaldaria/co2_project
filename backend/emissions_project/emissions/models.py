# emissions/models.py
from django.db import models


class Countries(models.Model):
    country_name = models.CharField(max_length=100)

    def __str__(self):
        return self.country_name

class Year(models.Model):
    Year = models.IntegerField()

    def __str__(self):
        return str(self.Year)

class Emissions(models.Model):
    country = models.ForeignKey(Countries, on_delete=models.CASCADE)
    year = models.ForeignKey(Year, on_delete=models.CASCADE)
    annual_co2_emissions = models.FloatField()

    def __str__(self):
        return f'{self.country.country_name} {self.year.Year}: {self.annual_co2_emissions}'

class ForecastResults(models.Model):
    date = models.DateTimeField()
    predicted_emissions = models.FloatField()

    def __str__(self):
        return f'Forecast for {self.date}: {self.predicted_emissions}'
