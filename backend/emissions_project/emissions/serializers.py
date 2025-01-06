# emissions/serializers.py
from rest_framework import serializers
from .models import Emissions, ForecastResults, Countries

class CountriesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Countries
        fields = '__all__'

class EmissionsSerializer(serializers.ModelSerializer):
    country = CountriesSerializer()

    class Meta:
        model = Emissions
        fields = '__all__'

class ForecastResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResults
        fields = '__all__'
