"""
Geographic location management for Open-Meteograms.

Provides geocoding via Nominatim and location metadata (elevation, timezone).
"""

import datetime as dt
import requests
import pytz
from timezonefinder import TimezoneFinder


def nominatim(location: str) -> list:
    """Geocode a location string using OpenStreetMap Nominatim."""
    url = 'https://nominatim.openstreetmap.org/search?'
    headers = {"User-Agent": f'Spotweather_{location}'}
    params = {
        'format': 'geocodejson',
        'namedetails': 1,
        'addressdetails': 1,
        'accept-language': 'en-US,en;q=0.8,es-ES,es;q=0.9',
        'q': location
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    results = response.json()['features']
    for index, result in enumerate(results):
        print(index, result['properties']['geocoding']['label'])
    return results


class Place:
    """Geographic location with elevation, timezone and utility URLs."""

    def __init__(self, location: dict):
        self.properties = self._properties(location)

        _label = self.properties.get('name', '')
        _city  = self.properties.get('city')
        self.name = _city if _city else ','.join(_label.split(',')[:2]).strip()

        coords = location.get('geometry', {}).get('coordinates', [None, None])
        self.lon, self.lat = coords

        self.elev = self.elevation() if self.lat and self.lon else None

        self.map = (
            f'https://www.google.com/maps/@?api=1&map_action=map'
            f'&basemap=satellite&center={self.lat}%2C{self.lon}'
            if self.lat and self.lon else None
        )

        self.tzinfo, self.delta_time = self.time_zone()
        self.meteo = self.meteo_url()

    def _properties(self, location: dict):
        properties = {}

        geocoding = location.get('properties', {}).get('geocoding', {})
        admin = geocoding.get('admin', {}) or {}

        properties['name'] = geocoding.get('label') or geocoding.get('name')

        properties['type'] = geocoding.get('type')
        city = geocoding.get('city')
        if not city and geocoding.get('type') in (
                'city', 'town', 'village', 'municipality', 'hamlet'):
            city = geocoding.get('name')
        properties['city'] = city
        properties['county'] = admin.get('level6')
        properties['state'] = geocoding.get('state')
        properties['country'] = geocoding.get('country')
        properties['country_code'] = geocoding.get('country_code')

        return properties

    def elevation(self):
        try:
            url = f'https://api.open-meteo.com/v1/elevation?latitude={self.lat}&longitude={self.lon}'
            response = requests.get(url, timeout=5).json()
            return round(response['elevation'][0])
        except Exception:
            return None

    def time_zone(self):
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lat=self.lat, lng=self.lon)
        if not tz_str:
            return pytz.UTC, 0
        tzinfo = pytz.timezone(tz_str)
        delta_time = dt.datetime.now(tzinfo).hour - dt.datetime.now().hour
        return tzinfo, delta_time

    def meteo_url(self):
        return {
            'windy': f'https://www.windy.com/{self.lat}/{self.lon}/wind?{self.lat},{self.lon},10',
            'meteoblue': f'https://www.meteoblue.com/es/tiempo/pronostico/multimodelensemble/{self.lat}N{self.lon}E',
        }
