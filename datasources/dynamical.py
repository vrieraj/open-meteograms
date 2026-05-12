"""Dynamical.org vertical forecast datasource."""

from __future__ import annotations

import os
import requests
import numpy as np
import pandas as pd

from core.vertical_dataset import VerticalDataset

DYNAMICAL_API_URL = os.getenv('DYNAMICAL_API_URL', 'https://api.dynamical.org/v1/forecast/sounding')
SUPPORTED_MODELS = {
    'ECMWF': 'ecmwf',
    'IFS': 'ecmwf',
    'GFS': 'gfs',
}


def _dewpoint_from_t_rh(t, rh):
    e_s = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = (rh / 100.0) * e_s
    return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))


def fetch_dynamical_forecast(model, lat, lon, init_time, forecast_hour):
    model_key = SUPPORTED_MODELS.get(str(model).upper())
    if model_key is None:
        raise ValueError(f"Unsupported dynamical model '{model}'. Use ECMWF/IFS or GFS.")

    token = os.getenv('DYNAMICAL_API_TOKEN')
    if not token:
        raise RuntimeError('DYNAMICAL_API_TOKEN is required to query dynamical.org')

    headers = {'Authorization': f'Bearer {token}'}
    params = {
        'model': model_key,
        'lat': float(lat),
        'lon': float(lon),
        'init_time': str(init_time),
        'forecast_hour': int(forecast_hour),
    }

    response = requests.get(DYNAMICAL_API_URL, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    rows = payload.get('data', [])
    if not rows:
        raise ValueError('Dynamical API returned no sounding rows')

    df = pd.DataFrame(rows)
    rename = {
        'valid_time': 'time',
        'level': 'pressure',
        'rh': 'relative_humidity',
        'u_wind': 'u',
        'v_wind': 'v',
        'z': 'geopotential_height',
    }
    df = df.rename(columns=rename)

    required = {'time', 'pressure', 'temperature', 'relative_humidity', 'u', 'v', 'geopotential_height'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Dynamical response missing required fields: {sorted(missing)}')

    df['wind_speed'] = np.hypot(df['u'], df['v'])
    df['wind_direction'] = (270 - np.degrees(np.arctan2(df['v'], df['u']))) % 360
    df['dewpoint'] = _dewpoint_from_t_rh(df['temperature'], df['relative_humidity'])

    keep = [
        'time', 'pressure', 'temperature', 'relative_humidity', 'dewpoint',
        'u', 'v', 'wind_speed', 'wind_direction', 'geopotential_height',
    ]
    out = VerticalDataset(df[keep].copy())
    out.validate()
    return out
