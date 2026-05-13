"""ERA5 ARCO datasource producing a VerticalDataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.vertical_dataset import VerticalDataset

ARCO_DATASET = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'


def fetch_era5_arco(lat, lon, start_date, end_date, levels=None, variables=None):
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError('xarray is required for ERA5 ARCO backend') from exc

    levels = levels or [1000, 925, 850, 700, 500]
    variables = variables or [
        'temperature',
        'specific_humidity',
        'u_component_of_wind',
        'v_component_of_wind',
        'geopotential',
    ]

    # Public bucket: force anonymous access to avoid requiring local GCP ADC
    ds = xr.open_zarr(
        ARCO_DATASET,
        consolidated=False,
        storage_options={'token': 'anon'},
    )
    sub = ds[variables].sel(
        latitude=lat,
        longitude=lon,
        method='nearest',
    ).sel(
        time=slice(start_date, end_date),
        level=levels,
    )

    df = sub.to_dataframe().reset_index().rename(columns={'level': 'pressure'})
    df['temperature'] = df['temperature'] - 273.15
    df['geopotential_height'] = df['geopotential'] / 9.80665

    # Compute RH (%) from q, T and p (ERA5 ARCO provides specific_humidity)
    p = pd.to_numeric(df['pressure'], errors='coerce') * 100.0  # hPa -> Pa
    q = pd.to_numeric(df['specific_humidity'], errors='coerce')
    # vapor pressure from specific humidity
    e = (q * p) / np.clip(0.622 + 0.378 * q, 1e-9, None)
    # saturation vapor pressure (Pa), Bolton (1980)
    es = 611.2 * np.exp((17.67 * df['temperature']) / (df['temperature'] + 243.5))
    rh = 100.0 * e / np.clip(es, 1e-9, None)
    df['relative_humidity'] = np.clip(rh, 0.0, 100.0)

    ws = np.hypot(df['u_component_of_wind'], df['v_component_of_wind'])
    df['wind_speed'] = ws
    df['wind_direction'] = (270 - np.degrees(np.arctan2(df['v_component_of_wind'], df['u_component_of_wind']))) % 360

    e_s = 6.112 * np.exp(17.67 * df['temperature'] / (df['temperature'] + 243.5))
    e = (df['relative_humidity'] / 100.0) * e_s
    df['dewpoint'] = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

    df = df.rename(columns={
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
    })

    keep = [
        'time', 'pressure', 'temperature', 'relative_humidity', 'dewpoint',
        'u', 'v', 'wind_speed', 'wind_direction', 'geopotential_height'
    ]
    out = VerticalDataset(df[keep].copy())
    out.validate()
    return out
