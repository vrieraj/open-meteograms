"""
Open-Meteo API connector.

Handles HTTP requests to Open-Meteo forecast and archive endpoints.
Returns raw JSON responses; normalization is handled by MeteoSfc / MeteoVrt.

Reference: https://open-meteo.com/en/docs
"""

import requests

# ── Surface variables ─────────────────────────────────────────────────────

SFC_VARIABLES = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'vapour_pressure_deficit', 'is_day'
]


def fetch_surface(lat, lon, elev, tzinfo, fechas, modelo):
    """
    Fetch hourly surface weather data from Open-Meteo.

    Parameters
    ----------
    lat, lon : float — coordinates
    elev : float — elevation (m)
    tzinfo : pytz.timezone — IANA timezone
    fechas : list[str] — ['YYYY-MM-DD', 'YYYY-MM-DD']
    modelo : dict — from WEATHER_MODELS (must have 'type' and 'keyword')

    Returns
    -------
    dict : JSON response with 'hourly' key, or None on error
    """
    tipo = 'api' if modelo['type'] == 'forecast' else 'archive-api'
    url = f'https://{tipo}.open-meteo.com/v1/{modelo["type"]}'
    params = dict(
        latitude=lat,
        longitude=lon,
        elevation=elev,
        hourly=",".join(SFC_VARIABLES),
        timezone=str(tzinfo),
        start_date=fechas[0],
        end_date=fechas[1],
        models=modelo["keyword"]
    )
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    return response.json()


# ── Vertical / pressure level variables ───────────────────────────────────

def _build_vertical_vars(levels):
    """Build the hourly variable list for pressure-level requests."""
    vert_vars = ['boundary_layer_height']
    for level in levels:
        vert_vars.append(f'temperature_{level}hPa')
        vert_vars.append(f'relative_humidity_{level}hPa')
        vert_vars.append(f'wind_speed_{level}hPa')
        vert_vars.append(f'wind_direction_{level}hPa')
        vert_vars.append(f'geopotential_height_{level}hPa')
    return vert_vars


def fetch_vertical(lat, lon, elev, tzinfo, fechas, modelo, levels):
    """
    Fetch hourly pressure-level data from Open-Meteo.

    Parameters
    ----------
    lat, lon, elev, tzinfo, fechas, modelo : same as fetch_surface
    levels : list[int] — pressure levels in hPa (e.g. [1000, 975, ..., 700])

    Returns
    -------
    dict : JSON response with 'hourly' key, or None on error
    """
    vert_vars = _build_vertical_vars(levels)
    tipo = 'api' if modelo['type'] == 'forecast' else 'archive-api'
    url = f'https://{tipo}.open-meteo.com/v1/{modelo["type"]}'
    params = dict(
        latitude=lat,
        longitude=lon,
        elevation=elev,
        hourly=",".join(vert_vars),
        timezone=str(tzinfo),
        start_date=fechas[0],
        end_date=fechas[1],
        models=modelo["keyword"]
    )
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    return response.json()
