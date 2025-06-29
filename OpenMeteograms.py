import datetime as dt
import numpy as np
import pandas as pd
import pytz
import requests
from timezonefinder import TimezoneFinder

import matplotlib.pyplot as plt
import seaborn as sns

#### LOCATION REQUESTS ####

def nominatim(location:str) -> list:
    url = 'https://nominatim.openstreetmap.org/search?'
    headers = {"User-Agent": f'Spotweather_{location}'}
    params = {
        'format':'geocodejson',
        'namedetails':1,
        'addressdetails':1,
        'accept-language':'en-US,en;q=0.8,es-ES,es;q=0.9',
        'q':location
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
    def __init__(self, location:dict):
        self.properties = self.properties(location)
        self.name = self.properties['name']

        self.lon, self.lat = location['geometry']['coordinates']
        self.elev = self.elevation()
        self.map = f'https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite&center={self.lat}%2C{self.lon}'
        # URLs Google Maps  ->  https://developers.google.com/maps/documentation/urls/get-started?hl=es-419

        self.tzinfo, self.delta_time = self.time_zone()
        self.meteo = self.meteo_url()

    def properties(self, location:dict):
        properties = {}
        geocoding = location['properties']['geocoding']
        for feature in ['name', 'type', 'city','county','state', 'country','country_code']:
            properties[feature] = geocoding.get(feature, None) if feature != 'county' else geocoding['admin'].get('level6', None)
        return properties
    
    def elevation(self):
        url = f'https://api.open-meteo.com/v1/elevation?latitude={self.lat}&longitude={self.lon}'
        response = requests.get(url).json()
        elevation = round(response['elevation'][0])
        return elevation

    def time_zone(self):
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lat=self.lat, lng=self.lon)

        tzinfo = pytz.timezone(tz_str)
        delta_time = dt.datetime.now(tzinfo).hour - dt.datetime.now().hour

        return tzinfo, delta_time
    
    def meteo_url(self):
        meteo_url = {
            'windy' : f'https://www.windy.com/{self.lat}/{self.lon}/wind?{self.lat},{self.lon},10',
            'meteoblue' : f'https://www.meteoblue.com/es/tiempo/pronostico/multimodelensemble/{self.lat}N{self.lon}E',
            'ecmwf-meteogram' : self.ecmwf_url('opencharts_meteogram')
            #,'ecmwf-sounding' : self.ecmwf_url('opencharts_vertical-profile-meteogram') 
        }

        return meteo_url
    
    def ecmwf_url(self, product:str):
        url = f'https://charts.ecmwf.int/opencharts-api/v1/products/{product}'
        params = dict(
            base_time = dt.date.today().strftime('%Y-%m-%dT%H:%M:%SZ'),
            lat = self.lat,
            lon = self.lon
        )

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()['data']['link']['href']
    
#### OPEN-METEO DATA REQUESTS ####

weather_models = {
    '': {
        "provider": '',
        "keyword": '',
        "country": '',
        "resolution": '',
        "days": '',
        "frequency": '',
        "type": ''
    },
    "ICON": {
        "provider": "Deutscher Wetterdienst (DWD)",
        "keyword": "icon_seamless",
        "country": "Germany",
        "resolution": "2 - 11 km",
        "days": "7.5 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "GFS": {
        "provider": "NOAA",
        "keyword": "gfs_seamless",
        "country": "United States",
        "resolution": "3 - 25 km",
        "days": "16 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "AROME": {
        "provider": "MeteoFrance",
        "keyword": "meteofrance_seamless",
        "country": "France",
        "resolution": "1 - 25 km",
        "days": "4 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "IFS": {
        "provider": "ECMWF",
        "keyword": "ecmwf_ifs025",
        "country": "European Union",
        "resolution": "25 km",
        "days": "7 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "GSM JMA": {
        "provider": "JMA",
        "keyword": "jma_seamless",
        "country": "Japan",
        "resolution": "5 - 55 km",
        "days": "11 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "MET Nordic": {
        "provider": "MET Norway",
        "keyword": "metno_nordic",
        "country": "Norway",
        "resolution": "1 km",
        "days": "2.5 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "GEM": {
        "provider": "Canadian Weather Service",
        "keyword": "gem_seamless",
        "country": "Canada",
        "resolution": "2.5 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "GFS GRAPES": {
        "provider": "China Meteorological Administration (CMA)",
        "keyword": "cma_grapes_global",
        "country": "China",
        "resolution": "15 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "ACCESS-G": {
        "provider": "Australian Bureau of Meteorology (BOM)",
        "keyword": "bom_access_global",
        "country": "Australia",
        "resolution": "15 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "COSMO": {
        "provider": "AM ARPAE ARPAP",
        "keyword": "arpae_cosmo_seamless",
        "country": "Italy",
        "resolution": "2 km",
        "days": "3 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "ECMWF IFS": {
        "provider": "ECMWF",
        "keyword": "ecmwf_ifs",
        "country": "European Union",
        "resolution": "9 km",
        "days": "2017 to present",
        "frequency": "Daily with 2 days delay",
        "type": "archive"
    },
    "ERA5": {
        "provider": "ECMWF",
        "keyword": "era5",
        "country": "European Union",
        "resolution": "11 km",
        "days": "1950 to present",
        "frequency": "Daily with 5 days delay",
        "type": "archive"
    }
}

def select_models(list_model) -> list[dict]:
    models_selected = {}
    for modelo in list_model:
        models_selected[modelo] = weather_models[modelo]
    return models_selected

class MeteoSfc:
    def __init__(self, location:Place, fechas:list):
        self.lat = location.lat
        self.lon = location.lon
        self.elev = location.elev
        self.tzinfo = location.tzinfo
        self.fechas = fechas
        self.datos = pd.DataFrame()

    def openmeteo_request(self, fechas:list, modelo:dict):
        variables = ['temperature_2m','relative_humidity_2m','dew_point_2m',
                    'wind_speed_10m','wind_direction_10m','wind_gusts_10m','is_day']
        tipo_meteo = 'api' if modelo['type'] == 'forecast' else 'archive-api'
        url = f'https://{tipo_meteo}.open-meteo.com/v1/{modelo['type']}?'
        params = dict(
            latitude={self.lat},
            longitude={self.lon},
            elevation={self.elev},
            hourly={",".join(variables)},
            timezone=self.tzinfo,
            start_date={fechas[0]},
            end_date={fechas[1]},
            models={modelo["keyword"]}
        )
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()

    def transform_data(self, datos:pd.DataFrame):
        def wind_arrows():
            cut_arrows = pd.cut(
                datos['wind_direction_10m'] % 360,
                bins=[-1, 23, 67, 112, 157, 202, 247, 292, 337, 361],
                labels=[
                    r'$\downarrow$',  # S
                    r'$\swarrow$',    # SW
                    r'$\leftarrow$',  # W
                    r'$\nwarrow$',    # NW
                    r'$\uparrow$',    # N
                    r'$\nearrow$',    # NE
                    r'$\rightarrow$', # E
                    r'$\searrow$',    # SE
                    r'$\downarrow$'   # S (duplicate)
                ],
                right=False,
                ordered=False
            )

            return cut_arrows

        def estimation_fuel_moisture():
            tabla_hcfm = {
                'dia': {
                    't10': [1,2,2,3,4,5,5,6,7,7,7,8,9,9,10,10,11,12,13,13,13],
                    't21': [1,2,2,3,4,5,5,6,6,7,7,8,8,9,9,10,11,12,12,12,13],
                    't32': [1,1,2,2,3,4,5,5,6,7,7,8,8,8,9,10,10,11,12,12,13],
                    't43': [1,1,2,2,3,4,4,5,6,7,7,8,8,8,9,10,10,11,12,12,13],
                    'tmax': [1,1,2,2,3,4,4,5,6,7,7,8,8,8,9,10,10,11,12,12,13]
                },
                'noche': {
                    't10': [1,2,3,4,5,6,7,8,9,9,11,11,12,13,14,16,18,21,24,25,25],
                    't21': [1,2,3,4,5,6,6,8,8,9,10,11,11,12,14,16,17,20,23,25,25],
                    't32': [1,2,3,4,4,5,6,7,8,9,10,10,11,12,13,15,17,20,23,25,25],
                    't43': [1,2,3,3,4,5,6,7,8,9,9,10,10,11,13,14,16,19,22,25,25],
                    'tmax': [1,2,2,3,4,5,6,6,8,8,9,9,10,11,12,14,16,19,21,24,25]
                }
            }

            def get_temperature_range_index(temp):
                if temp is None:    
                    return None
                elif temp < 10:
                    return 't10'
                elif temp < 21:
                    return 't21'
                elif temp < 32:
                    return 't32'
                elif temp < 43:
                    return 't43'
                else:
                    return 'tmax'

            time_periods = ['dia' if hour == 1 else 'noche' for hour in datos.is_day]
            temperature_ranges = [get_temperature_range_index(temp) if not np.isnan(temp) else None for temp in datos.temperature_2m]            
            humidity = [round(hum/5) if not np.isnan(hum) else None for hum in datos.relative_humidity_2m]
            hcfm_values = [tabla_hcfm[time_periods[i]][temperature_ranges[i]][humidity[i]] if temperature_ranges[i] and humidity[i] is not None else None for i in range(len(datos))]

            return hcfm_values

        def estimation_probignition():
            tabla_probig = [
                [90, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                [90, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                [100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10],
                [100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10, 10],
                [100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10],
                [100, 90, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10],
                [100, 100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 30, 20, 20, 20, 10]
            ]
            
            temp = [int(t / 5) if not np.isnan(t) else None for t in datos.temperature_2m]
            hum = [int(hcfm - 2) if not np.isnan(hcfm) else None for hcfm in datos.fuel_moisture]
            
            probig_values = []
            for t, h in zip(temp, hum):
                if h is not None and h > 15:
                    h = 15
                if t is not None and h is not None:
                    probig_values.append(tabla_probig[t][h])
                else:
                    probig_values.append(None)
            
            return probig_values

        datos.time = pd.to_datetime(datos.time)
        datos['wind_direction_arrow'] = wind_arrows()
        datos['fuel_moisture'] = estimation_fuel_moisture()
        datos['prob_ignition'] = estimation_probignition()

        return datos
    
    def get_data_models(self, modelos:dict[str, dict]) -> pd.DataFrame:
        for model_name, model_data in modelos.items():
            openmeteo_data = self.openmeteo_request(self.fechas, model_data)
            df = self.transform_data(pd.DataFrame(openmeteo_data['hourly']))
            df['model'] = model_name
            self.datos = pd.concat([self.datos, df],ignore_index=True)
        return self.datos

    def get_data_years(self, years:list) -> pd.DataFrame:
        year_now, month_init, day_init = self.fechas[0].split('-')
        year_now, month_end, day_end = self.fechas[1].split('-')
        for year in years:
            fechas = ['-'.join([str(year),month_init, day_init]), '-'.join([str(year),month_end, day_end])]
            openmeteo_data = self.openmeteo_request(fechas, weather_models['ERA5'])
            df = self.transform_data(pd.DataFrame(openmeteo_data['hourly']))
            df['year'] = year
            self.datos = pd.concat([self.datos, df],ignore_index=True)
        self.datos['year'] = self.datos['year'].fillna(year_now)
        return self.datos

#### METEOGRAMS PLOTS ####

class MeteoPlot:
    def __init__(self, sfc_data:dict[str, dict], fechas:list):
        inicio = sfc_data.time.dt.strftime('%Y-%m-%d') >= fechas[0]
        final = sfc_data.time.dt.strftime('%Y-%m-%d') <= fechas[1]

        self.datos = sfc_data.loc[(inicio) & (final)]
        self.fechas = [self.format_fecha(fecha, format = '%d-%b') for fecha in fechas]
        self.meteoplot()

    def format_fecha(self, fecha:str, format:str):
        year, month, day = map(int, fecha.split('-'))
        fecha = dt.datetime(year, month, day).strftime(format)
        return fecha

    def meteoplot(self) -> plt.Figure:
            colores = {
                        'green': ['darkolivegreen', 'green', 'springgreen', 'yellowgreen'],
                        'yellow': ['orange', 'gold', 'darkgoldenrod', 'yellow'],
                        'red': ['red', 'darkred', 'salmon', 'crimson'],
                        'grey': ['black', 'gray', 'lightgray', 'silver'],
                        'blue': ['navy', 'darkcyan', 'deepskyblue', 'paleturquoise']
                    }

            fig, ax = plt.subplots(3,1,figsize=(15,10), sharex=True)
            ax1 = ax[1].twinx()
            ax2 = ax[2].twinx()

            for model, index in zip(self.datos.model.unique(), range(0, len(colores))):
                datos = self.datos[self.datos.model == model]

                sns.lineplot(datos, x='time', y='wind_speed_10m', color=colores['green'][index], ax=ax[0])
                sns.lineplot(datos, x='time', y='wind_gusts_10m', color=colores['yellow'][index], linestyle='--', ax=ax[0])
                for i, row in datos.iterrows():
                    ax[0].text(row['time'], self.datos.wind_gusts_10m.max() + index*3, row['wind_direction_arrow'], 
                            fontsize=18, ha='center', va='bottom', color=colores['grey'][index])
                    
                sns.lineplot(datos, x='time', y='temperature_2m', color=colores['red'][index], ax=ax[1])
                sns.lineplot(datos, x='time', y='dew_point_2m', color=colores['grey'][index], linestyle='--', ax=ax[1])
                sns.lineplot(datos, x='time', y='relative_humidity_2m', color=colores['blue'][index], ax=ax1)

                sns.lineplot(datos, x='time', y='fuel_moisture', color=colores['red'][index], ax=ax[2])
                sns.lineplot(datos, x='time', y='prob_ignition', color=colores['yellow'][index], ax=ax2)
                    
            ax[0].fill_between(datos.time, 0, self.datos.wind_gusts_10m.max() + 10, where=datos.is_day == 0, alpha=0.3, color='lightblue')
            ax[0].set_ylim(0, self.datos.wind_gusts_10m.max() + 15)
            ax[0].set_ylabel('Wind speed and gusts (km/h)')

            ax1.fill_between(datos.time, 0,100, where=datos.is_day == 0, alpha=0.3, color='lightblue')
            ax[1].set_ylim(-5,45)
            ax[1].set_yticks(range(-5,45,5))
            ax[1].set_yticklabels(range(-5,45,5))
            ax[1].set_ylabel('Temperature and dewpoint (ºC)')
            ax[1].grid()

            ax1.set_ylim(0,100)
            ax1.set_yticks(range(0,100,10))
            ax1.set_yticklabels(range(0,100,10))
            ax1.set_ylabel('Humidity relative (%)')

            ax2.fill_between(datos.time, 0,100, where=datos.is_day == 0, alpha=0.3, color='lightblue')
            ax[2].set_ylim(0,25)
            ax[2].set_yticks(range(0,25,5))
            ax[2].set_yticklabels(range(0,25,5))
            ax[2].set_ylabel('Fuel moisture (%)')
            ax[2].grid()

            ax2.set_ylim(0,100)
            ax2.set_yticks(range(0,100,10))
            ax2.set_yticklabels(range(0,100,10))
            ax2.set_ylabel('Probability of ignition (%)')
            
            ax[2].set_xticks(datos.time.dt.date.unique())
            ax[2].set_xticklabels(datos.time.dt.strftime('%d-%b').unique())
            ax[2].set_xlabel('Source: Open-Meteo.com Weather API')

            fig.suptitle(f'Meteogram {', '.join(self.datos.model.unique())} | {self.fechas[0]} - {self.fechas[1]}',
                        fontsize=12, 
                        fontweight='bold')
            fig.tight_layout()

            return fig