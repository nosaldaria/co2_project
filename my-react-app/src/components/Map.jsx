// src/components/Map.js
import React, { useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// Вставте свій токен Mapbox сюди
mapboxgl.accessToken = 'pk.eyJ1IjoiZGFyeW5hbm9zYWwiLCJhIjoiY20ybGs2OWR5MGN2NDJrczlvNDY2ejR0MyJ9.DR29qpjTFFxBnjmbfWOSIQ';  // Замінити на свій токен

const MapComponent = () => {
    useEffect(() => {
        // Ініціалізація мапи
        const map = new mapboxgl.Map({
            container: 'map',  // Ідентифікатор контейнера
            style: 'mapbox://styles/mapbox/streets-v11',  // Стиль мапи
            center: [37.7749, -122.4194],  // Початкові координати
            zoom: 10,  // Початковий зум
        });

        return () => map.remove();  // Очистка після компонування
    }, []);

    return <div id="map" style={{ height: '100vh', width: '100%' }}></div>;
};

export default MapComponent;
