import datetime as dt
import numpy as np
import pandas as pd
import pytz
import requests
from timezonefinder import TimezoneFinder

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns

from weather_models import WEATHER_MODELS

#### LOCATION REQUESTS ####

def nominatim(location: str) -> list:
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
    def __init__(self, location: dict):
        self.properties = self._properties(location)
        self.name = self.properties['name']

        self.lon, self.lat = location['geometry']['coordinates']
        self.elev = self.elevation()
        self.map = f'https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite&center={self.lat}%2C{self.lon}'

        self.tzinfo, self.delta_time = self.time_zone()
        self.meteo = self.meteo_url()

    def _properties(self, location: dict):
        properties = {}
        geocoding = location['properties']['geocoding']
        for feature in ['name', 'type', 'city', 'county', 'state', 'country', 'country_code']:
            properties[feature] = geocoding.get(feature, None) if feature != 'county' else geocoding['admin'].get(
                'level6', None)
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
            'windy': f'https://www.windy.com/{self.lat}/{self.lon}/wind?{self.lat},{self.lon},10',
            'meteoblue': f'https://www.meteoblue.com/es/tiempo/pronostico/multimodelensemble/{self.lat}N{self.lon}E',
        }
        return meteo_url


#### OPEN-METEO DATA REQUESTS ####

class MeteoSfc:
    # Pressure levels for vertical profile
    VERTICAL_LEVELS_ALL = [1000, 975, 950, 925, 900, 850, 800, 700]  # 8 levels for data/BLH calc
    VERTICAL_LEVELS = [1000, 925, 850, 700]  # 4 levels for barb display
    VERTICAL_ALTITUDES = {
        1000: 110, 975: 320, 950: 500, 925: 800,
        900: 1000, 850: 1500, 800: 1900, 700: 3000
    }

    def __init__(self, location: Place, fechas: list):
        self.name = location.name
        self.lat = location.lat
        self.lon = location.lon
        self.elev = location.elev
        self.tzinfo = location.tzinfo
        self.fechas = fechas
        self.datos = pd.DataFrame()
        self.vertical = None  # Will hold vertical profile data if requested
        self.weather_models = WEATHER_MODELS.copy()

    def select_models(self, list_model) -> list[dict]:
        models_selected = {}
        for modelo in list_model:
            models_selected[modelo] = self.weather_models[modelo]
        return models_selected

    # ── Surface data request ──────────────────────────────────────────────

    def openmeteo_request(self, fechas: list, modelo: dict):
        variables = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
            'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
            'vapour_pressure_deficit', 'is_day'
        ]
        tipo_meteo = 'api' if modelo['type'] == 'forecast' else 'archive-api'
        url = f'https://{tipo_meteo}.open-meteo.com/v1/{modelo["type"]}'
        params = dict(
            latitude=self.lat,
            longitude=self.lon,
            elevation=self.elev,
            hourly=",".join(variables),
            timezone=str(self.tzinfo),
            start_date=fechas[0],
            end_date=fechas[1],
            models=modelo["keyword"]
        )
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()

    # ── Vertical profile data request ─────────────────────────────────────

    def request_vertical(self, modelo_name: str = 'IFS'):
        """
        Request vertical profile data for a single model.
        Accepts model names (e.g. 'IFS', 'GFS', 'AROME') or year strings
        (e.g. '2024') which automatically use ERA5 archive with date shifting.
        Requests 8 pressure levels for BLH calculation, displays barbs at 4.
        If the API does not provide boundary_layer_height, it is estimated
        using the Bulk Richardson Number method (Ri_crit=0.25).
        """
        # Detect if modelo_name is a year (ERA5 historical)
        is_year = modelo_name.isdigit() and 1950 <= int(modelo_name) <= 2100
        if is_year:
            year = int(modelo_name)
            modelo = self.weather_models['ERA5']
            year_now, month_init, day_init = self.fechas[0].split('-')
            _, month_end, day_end = self.fechas[1].split('-')
            fechas_vertical = [
                f'{year}-{month_init}-{day_init}',
                f'{year}-{month_end}-{day_end}'
            ]
        else:
            modelo = self.weather_models[modelo_name]
            fechas_vertical = self.fechas

        # Build variable list for 8 pressure levels + BLH
        vert_vars = ['boundary_layer_height']
        for level in self.VERTICAL_LEVELS_ALL:
            vert_vars.append(f'temperature_{level}hPa')
            vert_vars.append(f'relative_humidity_{level}hPa')
            vert_vars.append(f'wind_speed_{level}hPa')
            vert_vars.append(f'wind_direction_{level}hPa')
            vert_vars.append(f'geopotential_height_{level}hPa')

        tipo_meteo = 'api' if modelo['type'] == 'forecast' else 'archive-api'
        url = f'https://{tipo_meteo}.open-meteo.com/v1/{modelo["type"]}'
        params = dict(
            latitude=self.lat,
            longitude=self.lon,
            elevation=self.elev,
            hourly=",".join(vert_vars),
            timezone=str(self.tzinfo),
            start_date=fechas_vertical[0],
            end_date=fechas_vertical[1],
            models=modelo["keyword"]
        )
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None

        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])

        # If year mode, shift dates to current year (same as get_data_years)
        if is_year:
            delta = int(self.fechas[0].split('-')[0]) - year
            df.loc[:, 'time'] = df['time'] + pd.DateOffset(years=delta)

        # Coerce all numeric columns
        for level in self.VERTICAL_LEVELS_ALL:
            for var in ['temperature', 'relative_humidity', 'wind_speed',
                        'wind_direction', 'geopotential_height']:
                col = f'{var}_{level}hPa'
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute u,v wind components for barbs (knots) — display levels only
        for level in self.VERTICAL_LEVELS:
            ws = df[f'wind_speed_{level}hPa']
            wd = df[f'wind_direction_{level}hPa']
            ws_kt = ws * 0.539957  # km/h → knots
            df[f'u_{level}'] = -ws_kt * np.sin(np.radians(wd))
            df[f'v_{level}'] = -ws_kt * np.cos(np.radians(wd))

        # Detect inversions between display levels
        for i in range(len(self.VERTICAL_LEVELS) - 1):
            lower, upper = self.VERTICAL_LEVELS[i], self.VERTICAL_LEVELS[i + 1]
            t_lower = df[f'temperature_{lower}hPa']
            t_upper = df[f'temperature_{upper}hPa']
            df[f'inversion_{lower}_{upper}'] = t_upper > t_lower

        # ── BLH: use API value, fallback to Bulk Richardson calculation ──
        df['boundary_layer_height'] = pd.to_numeric(
            df.get('boundary_layer_height'), errors='coerce')

        blh_computed = self._compute_blh_richardson(df)
        df['blh_richardson'] = blh_computed

        # Fill missing API BLH with computed values
        mask_missing = df['boundary_layer_height'].isna()
        df.loc[mask_missing, 'boundary_layer_height'] = df.loc[mask_missing, 'blh_richardson']

        n_filled = mask_missing.sum()
        n_total = len(df)
        if n_filled > 0:
            print(f"  BLH: {n_total - n_filled}/{n_total} from API, "
                  f"{n_filled}/{n_total} estimated via Richardson (Ri=0.25)")

        self.vertical = df
        self.vertical_model = modelo_name
        return df

    def _compute_blh_richardson(self, df):
        """
        Estimate Boundary Layer Height using Bulk Richardson Number.

        Ri(z) = g * (θv(z) - θv_sfc) * (z - z_sfc) / [θv_sfc * ((u-u_sfc)² + (v-v_sfc)²)]

        When Ri crosses Ri_crit=0.25, linear interpolation gives BLH in meters.
        Uses virtual potential temperature θv to account for moisture.

        Reference: Seidel et al. (2012), Vogelezang & Holtslag (1996).
        ERA5 uses Ri_crit=0.25 (ECMWF IFS documentation).
        """
        g = 9.81
        Ri_crit = 0.25
        levels = self.VERTICAL_LEVELS_ALL  # 8 levels, low to high pressure

        blh_values = np.full(len(df), np.nan)

        for idx in range(len(df)):
            # ── Surface reference (1000 hPa) ──
            T_sfc = df[f'temperature_{levels[0]}hPa'].iat[idx]
            rh_sfc = df[f'relative_humidity_{levels[0]}hPa'].iat[idx]
            z_sfc = df[f'geopotential_height_{levels[0]}hPa'].iat[idx]
            ws_sfc = df[f'wind_speed_{levels[0]}hPa'].iat[idx]
            wd_sfc = df[f'wind_direction_{levels[0]}hPa'].iat[idx]

            if any(np.isnan(x) for x in [T_sfc, rh_sfc, z_sfc, ws_sfc, wd_sfc]):
                continue

            # Virtual potential temperature at surface
            theta_v_sfc = self._theta_v(T_sfc, rh_sfc, levels[0])

            # Wind components at surface (m/s)
            u_sfc = -ws_sfc / 3.6 * np.sin(np.radians(wd_sfc))
            v_sfc = -ws_sfc / 3.6 * np.cos(np.radians(wd_sfc))

            # ── Scan upward through levels ──
            Ri_prev = 0.0
            z_prev = z_sfc

            for lev in levels[1:]:
                T = df[f'temperature_{lev}hPa'].iat[idx]
                rh = df[f'relative_humidity_{lev}hPa'].iat[idx]
                z = df[f'geopotential_height_{lev}hPa'].iat[idx]
                ws = df[f'wind_speed_{lev}hPa'].iat[idx]
                wd = df[f'wind_direction_{lev}hPa'].iat[idx]

                if any(np.isnan(x) for x in [T, rh, z, ws, wd]):
                    continue

                theta_v = self._theta_v(T, rh, lev)
                dz = z - z_sfc
                if dz <= 0:
                    continue

                u = -ws / 3.6 * np.sin(np.radians(wd))
                v = -ws / 3.6 * np.cos(np.radians(wd))
                du2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2
                du2 = max(du2, 0.01)  # avoid division by zero

                Ri = g * (theta_v - theta_v_sfc) * dz / (theta_v_sfc * du2)

                if Ri >= Ri_crit:
                    # Linear interpolation between this level and previous
                    if Ri != Ri_prev:
                        frac = (Ri_crit - Ri_prev) / (Ri - Ri_prev)
                        blh_z = z_prev + frac * (z - z_prev) - z_sfc
                    else:
                        blh_z = dz
                    blh_values[idx] = max(blh_z, 10)  # minimum 10 m
                    break

                Ri_prev = Ri
                z_prev = z

        return blh_values

    @staticmethod
    def _theta_v(T_celsius, rh_pct, p_hPa):
        """
        Virtual potential temperature θv (K).

        θ  = T * (1000/p)^0.286           (potential temperature)
        e_s = 6.112 * exp(17.67*T / (T+243.5))  (saturation vapor pressure, hPa)
        r  = 0.622 * e / (p - e)           (mixing ratio, kg/kg)
        θv = θ * (1 + 0.61 * r)            (virtual correction)
        """
        T_K = T_celsius + 273.15
        theta = T_K * (1000.0 / p_hPa) ** 0.286
        e_s = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
        e = rh_pct / 100.0 * e_s
        r = 0.622 * e / max(p_hPa - e, 0.1)
        return theta * (1 + 0.61 * r)

    # ── Data transforms ───────────────────────────────────────────────────

    def transform_data(self, datos: pd.DataFrame) -> pd.DataFrame:
        datos = datos.copy(deep=True)

        def wind_arrows():
            cut_arrows = pd.cut(
                datos['wind_direction_10m'] % 360,
                bins=[-1, 23, 67, 112, 157, 202, 247, 292, 337, 361],
                labels=[
                    r'$\downarrow$', r'$\swarrow$', r'$\leftarrow$',
                    r'$\nwarrow$', r'$\uparrow$', r'$\nearrow$',
                    r'$\rightarrow$', r'$\searrow$', r'$\downarrow$'
                ],
                right=False, ordered=False
            )
            return cut_arrows

        def estimation_fuel_moisture_fosberg():
            """Fosberg Table A lookup — classic NFDRS field method."""
            tabla_hcfm = {
                'dia': {
                    't10': [1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 13],
                    't21': [1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 12, 12, 13],
                    't32': [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
                    't43': [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
                    'tmax': [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13]
                },
                'noche': {
                    't10': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 14, 16, 18, 21, 24, 25, 25],
                    't21': [1, 2, 3, 4, 5, 6, 6, 8, 8, 9, 10, 11, 11, 12, 14, 16, 17, 20, 23, 25, 25],
                    't32': [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 15, 17, 20, 23, 25, 25],
                    't43': [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 13, 14, 16, 19, 22, 25, 25],
                    'tmax': [1, 2, 2, 3, 4, 5, 6, 6, 8, 8, 9, 9, 10, 11, 12, 14, 16, 19, 21, 24, 25]
                }
            }

            def get_temp_key(temp):
                if temp is None or np.isnan(temp):
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

            hcfm_values = []
            for _, row in datos.iterrows():
                period = 'dia' if row['is_day'] == 1 else 'noche'
                temp_key = get_temp_key(row['temperature_2m'])
                if temp_key is None or pd.isna(row['relative_humidity_2m']):
                    hcfm_values.append(None)
                    continue
                hum_idx = int(np.clip(round(row['relative_humidity_2m'] / 5), 0, 20))
                hcfm_values.append(tabla_hcfm[period][temp_key][hum_idx])
            return hcfm_values

        def estimation_fuel_moisture_vpd():
            """Resco de Dios et al. (2015, 2024) — VPD-based semi-mechanistic model.
            FM = FM0 + FM1 * exp(-m * D)
            Calibrated on BONFIRE global dataset (1603 records).
            Returns fuel moisture in % (typically 4-30%).
            """
            vpd = datos.get('vapour_pressure_deficit')
            if vpd is None:
                return [None] * len(datos)
            FM0, FM1, m = 3.5, 28.0, 1.5
            fm = FM0 + FM1 * np.exp(-m * vpd)
            return fm.where(vpd.notna(), other=None).tolist()

        def estimation_probignition():
            """Probability of ignition lookup table.
            Uses temperature (rows, 5°C bins) and fuel moisture (columns).
            """
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
            probig_values = []
            for _, row in datos.iterrows():
                t = row['temperature_2m']
                fm = row['fuel_moisture']
                if pd.isna(t) or fm is None or pd.isna(fm):
                    probig_values.append(None)
                    continue
                t_idx = int(np.clip(t / 5, 0, 8))
                h_idx = int(np.clip(fm - 2, 0, 15))
                probig_values.append(tabla_probig[t_idx][h_idx])
            return probig_values

        datos.loc[:, 'time'] = pd.to_datetime(datos['time'])
        datos.loc[:, 'wind_direction_arrow'] = wind_arrows()
        datos.loc[:, 'fuel_moisture'] = estimation_fuel_moisture_fosberg()
        datos.loc[:, 'fuel_moisture_vpd'] = estimation_fuel_moisture_vpd()
        datos.loc[:, 'prob_ignition'] = estimation_probignition()

        return datos

    # ── Data retrieval methods ────────────────────────────────────────────

    def get_data_models(self, models: list) -> pd.DataFrame:
        modelos = self.select_models(models)
        for model_name, model_data in modelos.items():
            openmeteo_data = self.openmeteo_request(self.fechas, model_data)
            if openmeteo_data is None:
                continue
            df = self.transform_data(pd.DataFrame(openmeteo_data['hourly'])).copy()
            df.loc[:, 'model'] = model_name
            self.datos = pd.concat([self.datos, df], ignore_index=True)
        return self.datos

    def get_data_years(self, years: list) -> pd.DataFrame:
        year_now, month_init, day_init = self.fechas[0].split('-')
        year_now, month_end, day_end = self.fechas[1].split('-')

        for year in years:
            fechas = [
                '-'.join([str(year), month_init, day_init]),
                '-'.join([str(year), month_end, day_end])
            ]
            openmeteo_data = self.openmeteo_request(fechas, self.weather_models['ERA5'])
            if openmeteo_data is None:
                continue
            df = self.transform_data(pd.DataFrame(openmeteo_data['hourly'])).copy()
            df.loc[:, 'model'] = str(year)

            delta = int(year_now) - year
            df.loc[:, 'time'] = df['time'] + pd.DateOffset(years=delta)

            self.datos = pd.concat([self.datos, df], ignore_index=True)
        return self.datos

    # ── Plotting ──────────────────────────────────────────────────────────

    def meteoplot(self, fechas: list[str] = [], models: list[str] = []) -> plt.Figure:

        def get_partial_palette(cmap_name: str, start: float, stop: float, n_colors: int):
            cmap = plt.get_cmap(cmap_name)
            return [cmap(x) for x in np.linspace(start, stop, max(n_colors, 1))]

        def filter_data():
            fechas_local = fechas if fechas else self.fechas
            init_date = pd.to_datetime(fechas_local[0])
            end_date = pd.to_datetime(fechas_local[1])

            datos = self.datos.loc[
                (self.datos.time >= init_date) & (self.datos.time <= end_date)
                ]

            models_local = models if models else datos.model.unique()
            datos = datos[datos['model'].isin(models_local)]

            if datos.empty:
                raise ValueError(f'Dates out of range: {init_date.date()} | {end_date.date()}')

            invalid = set(models_local) - set(datos.model.unique())
            if invalid:
                raise ValueError(f'Unknown models: {invalid}')

            return init_date, end_date, datos, datos.model.unique()

        # ── FILTER ─────────────────────────────────────────
        init_date, end_date, datos, model_list = filter_data()

        # 👉 CONVERSIÓN CENTRALIZADA (CLAVE)
        datos_plot = datos.copy()
        datos_plot.loc[:, 'time_mpl'] = mdates.date2num(datos_plot['time'])

        datos_ref = datos_plot.loc[datos_plot.model == model_list[0]].copy()

        init_date_mpl = mdates.date2num(init_date)
        end_date_mpl = mdates.date2num(end_date)

        # ── FLAGS ─────────────────────────────────────────
        n_models = len(model_list)
        single_model = (n_models == 1)
        has_vertical = (self.vertical is not None and single_model)

        # ── FIGURE ─────────────────────────────────────────
        total_rows = 4
        height_ratios = [1.4, 1, 1, 1] if has_vertical else [0.6, 1, 1, 1]

        fig, ax = plt.subplots(
            total_rows, 1, figsize=(10, 10),
            gridspec_kw={'height_ratios': height_ratios}
        )

        TOP, WIND, TEMP, FUEL = 0, 1, 2, 3

        ax_rh = ax[TEMP].twinx()

        # Ensure primary axis (with legend) draws on top of twinx
        ax[TEMP].set_zorder(ax_rh.get_zorder() + 1)
        ax[TEMP].patch.set_visible(False)  # transparent so twinx content shows through

        # ── AXIS FORMAT ───────────────────────────────────
        for i in range(total_rows):
            ax[i].set_xlim(init_date_mpl, end_date_mpl)
            ax[i].xaxis.set_major_locator(mdates.DayLocator())
            if i < total_rows - 1:
                ax[i].tick_params(labelbottom=False)
            else:
                ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                ax[i].set_xlabel('Source: Open-Meteo.com Weather API')

        # ── NIGHT SHADING ─────────────────────────────────
        mask = (datos_ref.is_day == 0).fillna(False)

        for a in [ax[WIND], ax[TEMP], ax[FUEL]]:
            a.fill_between(
                datos_ref['time_mpl'], 0, 100,
                where=mask,
                alpha=0.1, color='lightblue', zorder=0
            )

        # ── COLOR PALETTES (thematic) ─────────────────────
        colors = {
            'greens':  get_partial_palette('Greens_r',  0.1, 0.7, n_models),
            'yellows': get_partial_palette('Wistia_r',  0.1, 0.7, n_models),
            'reds':    get_partial_palette('Reds_r',    0.1, 0.7, n_models),
            'blues':   get_partial_palette('Blues_r',   0.1, 0.7, n_models),
            'greys':   get_partial_palette('Greys_r',   0.1, 0.7, n_models),
        }

        # ── VARIABLE SETTINGS ─────────────────────────────
        VAR_CFG = {
            'wind_speed_10m':       {'color': colors['greens'][0], 'palette': colors['greens'],
                                     'style': '-',  'label': 'Wind Speed'},
            'wind_gusts_10m':       {'color': colors['yellows'][0], 'palette': colors['yellows'],
                                     'style': '--', 'label': 'Wind Gusts'},
            'temperature_2m':       {'color': colors['reds'][0],   'palette': colors['reds'],
                                     'style': '-',  'label': 'Temperature'},
            'dew_point_2m':         {'color': colors['greys'][0],  'palette': colors['greys'],
                                     'style': '--', 'label': 'Dew Point'},
            'relative_humidity_2m': {'color': colors['blues'][0],  'palette': colors['blues'],
                                     'style': '-',  'label': 'Relative Humidity'},
            'fuel_moisture':        {'color': colors['yellows'][0], 'palette': colors['yellows'],
                                     'style': '--', 'label': 'FM Fosberg (1h)'},
            'fuel_moisture_vpd':    {'color': colors['reds'][0],   'palette': colors['reds'],
                                     'style': '-',  'label': 'FM VPD (Resco de Dios)'},
        }

        import matplotlib.lines as mlines

        def plot_var(feat, axis, data):
            """Plot a variable with proper color/label/style."""
            cfg = VAR_CFG[feat]
            if single_model:
                d = data[data.model == model_list[0]]
                axis.plot(d['time_mpl'], d[feat],
                          color=cfg['color'], linestyle=cfg['style'],
                          label=cfg['label'], zorder=5)
            else:
                for idx, model in enumerate(model_list):
                    d = data[data.model == model]
                    axis.plot(d['time_mpl'], d[feat],
                              color=cfg['palette'][idx], linestyle=cfg['style'],
                              label=model, zorder=5)

        # ── PANEL 0: VERTICAL or WIND DIRECTION ──────────
        if has_vertical:
            self._plot_vertical_profile(ax[TOP], datos_ref, init_date, end_date)
        else:
            self._plot_wind_direction(ax[TOP], datos_plot, model_list, datos_ref)

        # ── PANEL 1: WIND SPEED + GUSTS (single Y axis) ─
        plot_var('wind_speed_10m', ax[WIND], datos_plot)
        plot_var('wind_gusts_10m', ax[WIND], datos_plot)

        ax[WIND].set_ylabel('Wind (km/h)')
        ax[WIND].grid(True, alpha=0.3)

        # Dynamic Y: 0-60 default, auto if gusts > 50
        max_gusts = datos_plot['wind_gusts_10m'].max()
        if max_gusts <= 50:
            ax[WIND].set_ylim(0, 60)
        else:
            ax[WIND].set_ylim(0, None)  # auto upper

        if single_model:
            ax[WIND].legend(loc='upper left', fontsize=7)
        else:
            model_handles = [mlines.Line2D([], [], color=colors['greens'][i], label=m)
                             for i, m in enumerate(model_list)]
            style_handles = [
                mlines.Line2D([], [], color='grey', linestyle='-', label='Speed'),
                mlines.Line2D([], [], color='grey', linestyle='--', label='Gusts'),
            ]
            ax[WIND].legend(handles=model_handles + style_handles,
                            loc='upper left', fontsize=7, ncol=2)

        # ── PANEL 2: TEMP + DEWPOINT + RH ────────────────
        plot_var('temperature_2m', ax[TEMP], datos_plot)
        plot_var('dew_point_2m', ax[TEMP], datos_plot)
        plot_var('relative_humidity_2m', ax_rh, datos_plot)

        ax[TEMP].grid(True, alpha=0.3)
        ax[TEMP].set_ylabel('Temperature / Dewpoint (°C)')
        ax_rh.set_ylabel('Relative Humidity (%)')
        ax_rh.set_ylim(0, 100)
        ax_rh.set_yticks(range(0, 100, 10))

        # Dynamic Y: -5 to 40 default, auto if out of range
        temp_max = datos_plot['temperature_2m'].max()
        temp_min = datos_plot[['temperature_2m', 'dew_point_2m']].min().min()
        if temp_min >= -5 and temp_max <= 40:
            ax[TEMP].set_ylim(-5, 40)
            ax[TEMP].set_yticks(range(-5, 45, 5))
        # else: auto

        if single_model:
            # Left legend: temp + dewpoint
            h1, l1 = ax[TEMP].get_legend_handles_labels()
            ax[TEMP].legend(h1, l1, loc='upper left', fontsize=7)
            # Right legend: RH
            h2, l2 = ax_rh.get_legend_handles_labels()
            ax_rh.legend(h2, l2, loc='upper right', fontsize=7)
        else:
            # Left legend: models colored by temp palette + line styles
            model_handles_t = [mlines.Line2D([], [], color=colors['reds'][i], label=m)
                               for i, m in enumerate(model_list)]
            style_handles_t = [
                mlines.Line2D([], [], color='grey', linestyle='-', label='Temp'),
                mlines.Line2D([], [], color='grey', linestyle='--', label='Dewpoint'),
            ]
            ax[TEMP].legend(handles=model_handles_t + style_handles_t,
                            loc='upper left', fontsize=7, ncol=2)
            # Right legend: models colored by RH palette
            model_handles_rh = [mlines.Line2D([], [], color=colors['blues'][i], label=m)
                                for i, m in enumerate(model_list)]
            ax_rh.legend(handles=model_handles_rh, loc='upper right', fontsize=7)

        # ── PANEL 3: FUEL MOISTURE + IGNITION ────────────
        for feat in ['fuel_moisture', 'fuel_moisture_vpd']:
            if datos_plot[feat].dropna().empty:
                continue
            plot_var(feat, ax[FUEL], datos_plot)

        ax[FUEL].set_ylabel('Fuel Moisture (%)')
        ax[FUEL].set_ylim(0, 35)
        ax[FUEL].set_yticks(range(0, 35, 5))
        ax[FUEL].grid(True, alpha=0.3)

        if single_model:
            ax[FUEL].legend(loc='upper left', fontsize=7)
        else:
            model_handles = [mlines.Line2D([], [], color=colors['yellows'][i], label=m)
                             for i, m in enumerate(model_list)]
            style_handles = [
                mlines.Line2D([], [], color='grey', linestyle='--', label='Fosberg'),
                mlines.Line2D([], [], color='grey', linestyle='-', label='VPD'),
            ]
            ax[FUEL].legend(handles=model_handles + style_handles,
                            loc='upper left', fontsize=7, ncol=2)

        self._plot_ignition_semaphore(ax[FUEL], datos_ref)

        # ── TITLE ────────────────────────────────────────
        fig.suptitle(
            f'Meteogram {self.name} | Lat: {round(self.lat, 4)} Lon: {round(self.lon, 4)}',
            fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12)
        return fig

    # ── Vertical profile subplot ──────────────────────────────────────────

    def _plot_vertical_profile(self, ax, datos_ref, init_date, end_date):
        """Time-height cross section with wind barbs, BLH and inversions."""
        vd = self.vertical.copy()
        vd['time_mpl'] = mdates.date2num(vd['time'])

        # Filter to date range
        vd = vd[(vd.time >= init_date) & (vd.time <= end_date)].copy()
        if vd.empty:
            ax.text(0.5, 0.5, 'No vertical data in range',
                    transform=ax.transAxes, ha='center', va='center')
            return

        levels = self.VERTICAL_LEVELS  # [1000, 925, 850, 700]
        times = vd['time'].values

        # ── Night shading ──
        ax.fill_between(
            datos_ref['time_mpl'], min(levels) - 50, max(levels) + 50,
            where=(datos_ref.is_day == 0).fillna(False),
            alpha=0.08, color='lightblue', zorder=0
        )

        # ── Inversion shading ──
        layer_colors = {
            f'inversion_{levels[0]}_{levels[1]}': ('royalblue', 0.2),  # surface inversion
            f'inversion_{levels[1]}_{levels[2]}': ('darkorange', 0.2),  # low elevated
            f'inversion_{levels[2]}_{levels[3]}': ('darkorange', 0.15),  # high elevated
        }
        for i in range(len(levels) - 1):
            lower, upper = levels[i], levels[i + 1]
            col_name = f'inversion_{lower}_{upper}'
            color, alpha = layer_colors[col_name]
            inv_mask = vd[col_name].fillna(False).values.astype(bool)
            ax.fill_between(
                vd['time_mpl'], lower, upper,
                where=inv_mask,
                color=color, alpha=alpha, zorder=1,
                label=f'Inv. {lower}-{upper}' if i == 0 else None
            )

        # ── Wind barbs ──
        n_hours = len(vd)
        thin = max(1, n_hours // 40)
        vd_thin = vd.iloc[::thin]

        for level in levels:
            u = pd.to_numeric(vd_thin[f'u_{level}'], errors='coerce').values
            v = pd.to_numeric(vd_thin[f'v_{level}'], errors='coerce').values
            t = vd_thin['time_mpl'].values
            valid = np.isfinite(u) & np.isfinite(v)
            if valid.any():
                p = np.full(valid.sum(), level, dtype=float)
                ax.barbs(t[valid], p, u[valid], v[valid],
                         length=5, linewidth=0.5,
                         barb_increments=dict(half=5, full=10, flag=50),
                         zorder=5)

        # ── Boundary layer height ──
        # Convert BLH (meters) to approximate pressure level for plotting
        # Simple approximation: p ≈ 1013 * exp(-z / 8500)
        blh_m = pd.to_numeric(vd['boundary_layer_height'], errors='coerce').values
        blh_p = np.where(np.isfinite(blh_m), 1013.25 * np.exp(-blh_m / 8500.0), np.nan)
        ax.plot(vd['time_mpl'], blh_p,
                color='red', linewidth=2, linestyle='--',
                label='Boundary Layer Height', zorder=6)

        # ── Axes config ──
        ax.set_ylim(max(levels) + 20, min(levels) - 20)  # inverted: high pressure at bottom
        ax.set_yticks(levels)
        ax.set_yticklabels([str(l) for l in levels])
        ax.set_ylabel('Pressure (hPa)')
        ax.grid(True, alpha=0.3)

        # Right axis with approximate altitudes
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(levels)
        ax2.set_yticklabels([f'{self.VERTICAL_ALTITUDES[l]} m' for l in levels])
        ax2.set_ylabel('Altitude (m ASL)')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=7, ncol=2).set_zorder(10)

        ax.set_title(f'Vertical Profile — {self.vertical_model}', fontsize=9, loc='left')

    # ── Wind direction arrows subplot (multi-model) ───────────────────────

    def _plot_wind_direction(self, ax, datos, model_list, datos_ref):
        """Wind direction arrows panel for multi-model comparison."""
        ax.fill_between(
            datos_ref['time_mpl'], 0, 100,
            where=(datos_ref.is_day == 0).fillna(False),
            alpha=0.2, color='lightblue', zorder=0
        )
        for index, model in enumerate(model_list):
            datos_model = datos.loc[datos.model == model].copy()
            for _, row in datos_model.iterrows():
                ax.text(mdates.date2num(row['time']), index + 1, row['wind_direction_arrow'],
                        fontsize=18, ha='center', va='center', color='k', zorder=5)

        ax.set_ylim(0, len(model_list) + 1)
        ax.set_yticks(range(1, len(model_list) + 1))
        ax.set_yticklabels(model_list)

    # ── Ignition probability semaphore ────────────────────────────────────

    def _plot_ignition_semaphore(self, ax_fuel, datos_ref):
        """Color bar below the fuel moisture panel showing ignition probability."""
        prob = datos_ref['prob_ignition'].values
        times = mdates.date2num(pd.to_datetime(datos_ref['time']))

        if all(p is None for p in prob):
            return

        # Create color-mapped segments at the bottom of the fuel panel
        ylim = ax_fuel.get_ylim()
        bar_height = (ylim[1] - ylim[0]) * 0.06
        bar_bottom = ylim[0]

        # Semaphore colormap: green (safe) → yellow → orange → red (critical)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'ignition', ['#2d9e2d', '#f0c929', '#e88a1a', '#d92525'], N=256
        )
        norm = mcolors.Normalize(vmin=0, vmax=100)

        for i in range(len(times) - 1):
            p = prob[i]
            if p is None or np.isnan(p):
                continue
            color = cmap(norm(p))
            ax_fuel.axvspan(
                times[i], times[i + 1],
                ymin=0, ymax=bar_height / (ylim[1] - ylim[0]),
                color=color, alpha=0.8, zorder=4
            )

        # Adjust ylim to make room for the bar
        ax_fuel.set_ylim(bar_bottom - bar_height * 0.5, ylim[1])

        # Add label
        ax_fuel.text(
            times[0], bar_bottom + bar_height * 0.3,
            'Ignition Prob.', fontsize=7, va='center', ha='left',
            color='white', fontweight='bold', zorder=5
        )