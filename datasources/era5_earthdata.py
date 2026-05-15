"""ERA5 pressure-level datasource via EarthDataHub DestinE Zarr backend."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from core.vertical_dataset import VerticalDataset

EARTHDATAHUB_URL = (
    'https://data.earthdatahub.destine.eu/'
    'era5/reanalysis-era5-pressure-levels-v0.zarr'
)

# Map legacy/ARCO-style variable names -> EarthDataHub short names
VARIABLE_MAP = {
    'temperature': 't',
    'specific_humidity': 'q',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'geopotential': 'z',
    'relative_humidity': 'r',
}

# Standard ERA5 pressure levels available in EarthDataHub
AVAILABLE_LEVELS = {1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
                    150, 100, 70, 50, 30, 20, 10, 5, 1}

# Module-level cache for the Zarr dataset (expensive to open)
_DS_CACHE = None


def _get_dataset():
    global _DS_CACHE
    if _DS_CACHE is not None:
        return _DS_CACHE
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError('xarray is required for ERA5 EarthDataHub backend') from exc

    token = os.environ.get('EARTHDATAHUB_API_KEY', '')
    if not token:
        raise ValueError(
            "EARTHDATAHUB_API_KEY not set. Add to .env file:\n"
            "  EARTHDATAHUB_API_KEY=your_token"
        )

    url = f'https://edh:{token}@data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr'
    ds = xr.open_dataset(url, chunks={}, engine='zarr')
    _DS_CACHE = ds
    return ds


def fetch_era5_earthdata(lat, lon, start_date, end_date, levels=None, variables=None):
    """
    Fetch ERA5 pressure-level data from EarthDataHub DestinE Zarr.

    Parameters
    ----------
    lat, lon : float — coordinates (lon in -180..180 or 0..360)
    start_date, end_date : str — ISO datetime range
    levels : list[int], optional — pressure levels in hPa
              (default [1000, 925, 850, 700, 500])
    variables : list[str], optional — legacy/ARCO-style variable names

    Returns
    -------
    VerticalDataset
    """
    levels = levels or [1000, 925, 850, 700, 500]

    # EarthDataHub uses 0-360 longitude
    lon = lon % 360

    ds = _get_dataset()
    available_vars = set(ds.data_vars)

    # Map requested variable names to EarthDataHub short names
    var_names = variables or [
        'temperature', 'specific_humidity',
        'u_component_of_wind', 'v_component_of_wind', 'geopotential',
    ]
    edh_vars = []
    for v in var_names:
        mapped = VARIABLE_MAP.get(v, v)
        if mapped in available_vars:
            edh_vars.append(mapped)

    # Always include relative humidity for dewpoint computation
    if 'r' in available_vars and 'r' not in edh_vars:
        edh_vars.append('r')

    required_core = {'t', 'u', 'v', 'z'}
    missing_core = required_core - set(edh_vars)
    if missing_core:
        raise KeyError(
            f"EarthDataHub dataset missing core variables: {missing_core}"
        )

    # Filter to levels that exist in this dataset
    valid_levels = sorted(set(levels) & AVAILABLE_LEVELS, reverse=True)
    missing_levels = sorted(set(levels) - AVAILABLE_LEVELS)
    if missing_levels:
        print(f"  Warning: ERA5 EarthDataHub missing levels {missing_levels}; "
              f"using {valid_levels}")
    if not valid_levels:
        raise ValueError(f"No valid levels in {levels} for EarthDataHub dataset")

    sub = ds[edh_vars].sel(
        latitude=lat,
        longitude=lon,
        method='nearest',
    ).sel(
        valid_time=slice(start_date, end_date),
        isobaricInhPa=valid_levels,
    )

    df = sub.load().to_dataframe().reset_index()
    df = df.rename(columns={
        'valid_time': 'time',
        'isobaricInhPa': 'pressure',
        't': 'temperature',
        'q': 'specific_humidity',
        'u': 'u',
        'v': 'v',
        'z': 'geopotential',
        'r': 'relative_humidity_raw',
    })

    # Temperature: K -> degC
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'] - 273.15

    # Geopotential: m2/s2 -> m
    if 'geopotential' in df.columns:
        df['geopotential_height'] = df['geopotential'] / 9.80665

    # Wind: u/v (m/s) -> wind_speed (km/h) and direction
    if 'u' in df.columns and 'v' in df.columns:
        ws_ms = np.hypot(df['u'], df['v'])
        df['wind_speed'] = ws_ms * 3.6  # m/s -> km/h
        df['wind_direction'] = (
            270 - np.degrees(np.arctan2(df['v'], df['u']))
        ) % 360

    # Relative humidity: r may be 0-1 or 0-100; handle NaNs safely
    if 'relative_humidity_raw' in df.columns:
        rh = df['relative_humidity_raw'].astype('float64').values
        if rh.size == 0:
            df['relative_humidity'] = np.nan
        else:
            max_rh = np.nanmax(rh)
            if not np.isnan(max_rh) and max_rh <= 1.0:
                rh = rh * 100.0
            df['relative_humidity'] = rh

    # Dewpoint from T (degC) and RH (%)
    if 'temperature' in df.columns and 'relative_humidity' in df.columns:
        T = df['temperature'].astype('float64')
        rh = df['relative_humidity'].astype('float64')
        e_s = 6.112 * np.exp(17.67 * T / (T + 243.5))
        e = (rh / 100.0) * e_s
        ratio = np.clip(e / 6.112, 1e-9, None)
        df['dewpoint'] = (
            243.5 * np.log(ratio) / (17.67 - np.log(ratio))
        )

    keep = [
        'time', 'pressure', 'temperature', 'relative_humidity', 'dewpoint',
        'u', 'v', 'wind_speed', 'wind_direction', 'geopotential_height',
    ]
    extra = [c for c in ['specific_humidity'] if c in df.columns]

    out = VerticalDataset(df[keep + extra].copy())
    out.validate()
    return out
