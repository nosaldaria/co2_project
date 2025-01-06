# emissions/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Emissions, ForecastResults, Countries
from .serializers import EmissionsSerializer, ForecastResultsSerializer

class EmissionsList(APIView):
    def get(self, request):
        emissions = Emissions.objects.all()
        serializer = EmissionsSerializer(emissions, many=True)
        return Response(serializer.data)

class EmissionsByCountry(APIView):
    def get(self, request, country_id):
        emissions = Emissions.objects.filter(country_id=country_id)
        serializer = EmissionsSerializer(emissions, many=True)
        return Response(serializer.data)

class ForecastResultsList(APIView):
    def get(self, request):
        forecasts = ForecastResults.objects.all()
        serializer = ForecastResultsSerializer(forecasts, many=True)
        return Response(serializer.data)
