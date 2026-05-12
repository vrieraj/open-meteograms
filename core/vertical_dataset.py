"""Standard long-format vertical atmospheric dataset container."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = [
    'time',
    'pressure',
    'temperature',
    'relative_humidity',
    'dewpoint',
    'u',
    'v',
    'wind_speed',
    'wind_direction',
    'geopotential_height',
]

OPTIONAL_COLUMNS = [
    'specific_humidity',
    'omega',
    'vertical_velocity',
]


@dataclass
class VerticalDataset:
    """Container for tidy vertical profile data."""

    df: pd.DataFrame

    def validate(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = self.df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isna().any():
            raise ValueError('Invalid timestamps in time column')

        numeric_cols = [
            'pressure', 'temperature', 'relative_humidity', 'dewpoint',
            'u', 'v', 'wind_speed', 'wind_direction', 'geopotential_height',
        ]
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors='coerce')
            if values.isna().all():
                raise ValueError(f'Column {col} is non-numeric or empty')

        for _, group in df.sort_values(['time', 'pressure']).groupby('time'):
            p = pd.to_numeric(group['pressure'], errors='coerce').dropna().values
            if len(p) > 1 and not ((p[1:] <= p[:-1]).all() or (p[1:] >= p[:-1]).all()):
                raise ValueError('Pressure must be monotonic per timestamp')

    def get_profile(self, time) -> pd.DataFrame:
        t = pd.to_datetime(time)
        idx = (pd.to_datetime(self.df['time']) - t).abs().idxmin()
        actual = self.df.loc[idx, 'time']
        return self.df[self.df['time'] == actual].sort_values('pressure', ascending=False).copy()

    def get_time_series(self, level: float) -> pd.DataFrame:
        g = self.df.copy()
        g['delta'] = (pd.to_numeric(g['pressure'], errors='coerce') - float(level)).abs()
        series = g.sort_values(['time', 'delta']).groupby('time', as_index=False).first()
        return series.drop(columns=['delta'])
