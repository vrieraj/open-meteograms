"""
Vertical meteorology: pressure levels, BLH, profiles and Skew-T.

Pipeline: get_data(source, **kwargs) → normalize → self.datos (DataFrame)
Transforms: Bulk Richardson BLH, virtual potential temperature θv,
            wind component u/v, inversion detection.
Plotting: profileplot() — standalone time-height cross section
          plot_on_axes()  — embeddable in MeteoSfc.meteoplot()
          skewt()         — single-time radiosonde (placeholder)

Reference:
    BLH Richardson: Seidel et al. (2012), Vogelezang & Holtslag (1996)
    ERA5 uses Ri_crit=0.25 (ECMWF IFS documentation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .weather_models import WEATHER_MODELS
from datasources.openmeteo import fetch_vertical
from datasources.dynamical import fetch_dynamical_forecast, SUPPORTED_MODELS
from datasources.era5_arco import fetch_era5_arco
from core.vertical_dataset import VerticalDataset


class MeteoVrt:
    """Vertical profile data and visualization."""

    # 4 levels for meteogram data pipeline, BLH calculation and barb display
    LEVELS_ALL = [1000, 925, 850, 700]
    LEVELS_DISPLAY = [1000, 925, 850, 700]

    # 10 levels for Skew-T (fetched on demand, separate from meteogram data)
    LEVELS_SKEWT = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500]

    # Approximate altitudes (m ASL)
    ALTITUDES = {
        1000: 110, 975: 320, 950: 500, 925: 800,
        900: 1000, 850: 1500, 800: 1900, 700: 3000,
        600: 4200, 500: 5600,
    }

    def __init__(self, place, fechas: list[str]):
        """
        Parameters
        ----------
        place : Place — geographic location
        fechas : list — ['YYYY-MM-DD', 'YYYY-MM-DD'] date range
        """
        self.name = place.name
        self.lat = place.lat
        self.lon = place.lon
        self.elev = place.elev
        self.tzinfo = place.tzinfo
        self.fechas = fechas
        self.datos = None
        self.dataset = None
        self.source_name = None
        self._datos_skewt = None    # lazy-loaded on first skewt() call
        self._skewt_modelo = None   # model dict used for Skew-T fetch
        self._skewt_fechas = None   # date range used for Skew-T fetch
        self.weather_models = WEATHER_MODELS.copy()


    @staticmethod
    def _dewpoint_from_t_rh(T, rh):
        e_s = 6.112 * np.exp(17.67 * T / (T + 243.5))
        e = (rh / 100.0) * e_s
        return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

    def _to_vertical_dataset(self, wide_df: pd.DataFrame, levels: list[int]) -> VerticalDataset:
        records = []
        for _, row in wide_df.iterrows():
            t = row['time']
            for level in levels:
                tcol = f'temperature_{level}hPa'
                rhcol = f'relative_humidity_{level}hPa'
                wcol = f'wind_speed_{level}hPa'
                dcol = f'wind_direction_{level}hPa'
                zcol = f'geopotential_height_{level}hPa'
                if tcol not in row.index:
                    continue
                T = row.get(tcol, np.nan)
                rh = row.get(rhcol, np.nan)
                ws = row.get(wcol, np.nan)
                wd = row.get(dcol, np.nan)
                z = row.get(zcol, np.nan)
                if pd.isna(T) or pd.isna(rh):
                    continue
                dew = self._dewpoint_from_t_rh(T, rh)
                ws_kt = ws * 0.539957 if not pd.isna(ws) else np.nan
                u = -ws_kt * np.sin(np.radians(wd)) if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan
                v = -ws_kt * np.cos(np.radians(wd)) if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan
                records.append({
                    'time': t, 'pressure': level, 'temperature': T,
                    'relative_humidity': rh, 'dewpoint': dew, 'u': u, 'v': v,
                    'wind_speed': ws, 'wind_direction': wd, 'geopotential_height': z,
                })
        ds = VerticalDataset(pd.DataFrame.from_records(records))
        ds.validate()
        return ds

    def get_profile(self, time):
        if self.dataset is None:
            raise ValueError('No vertical dataset. Call get_data() first.')
        return self.dataset.get_profile(time)

    def get_time_series(self, level):
        if self.dataset is None:
            raise ValueError('No vertical dataset. Call get_data() first.')
        return self.dataset.get_time_series(level)

    # ══════════════════════════════════════════════════════════════════════
    #  DATA PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    def get_data(self, source: str = 'openmeteo', **kwargs) -> pd.DataFrame:
        """
        Unified vertical data pipeline.

        Parameters
        ----------
        source : str
            'openmeteo' — NWP model (requires model='AROME')
            'era5'      — ERA5 by year (requires year=2024)
            'dynamical' — Dynamical.org soundings (ECMWF/GFS)
            Future: 'radiosonde' (requires station='08001')

        Returns
        -------
        pd.DataFrame — self.datos
        """
        if source == 'openmeteo':
            model = kwargs.get('model', 'IFS')
            self._fetch_openmeteo(model)

        elif source == 'era5':
            year = kwargs.get('year')
            if year is None:
                raise ValueError("'era5' source requires year=YYYY")
            self._fetch_era5_year(year)

        elif source == 'dynamical':
            model = kwargs.get('model', 'ECMWF')
            init_time = kwargs.get('init_time')
            forecast_hour = kwargs.get('forecast_hour', 0)
            if init_time is None:
                raise ValueError("'dynamical' source requires init_time='YYYY-MM-DDTHH:MM:SSZ'")
            self._fetch_dynamical(model, init_time, forecast_hour)

        else:
            raise ValueError(f"Unknown source: '{source}'. "
                             f"Available: 'openmeteo', 'era5', 'dynamical'")

        return self.datos

    def _fetch_openmeteo(self, model_name: str):
        """Fetch pressure-level data from Open-Meteo forecast model."""
        if model_name not in self.weather_models:
            raise ValueError(f"Unknown model: '{model_name}'")
        modelo = self.weather_models[model_name]
        data = fetch_vertical(self.lat, self.lon, self.elev,
                              self.tzinfo, self.fechas, modelo,
                              self.LEVELS_ALL)
        if data is None:
            return
        self._process_response(data, model_name)
        self._skewt_modelo = modelo
        self._skewt_fechas = self.fechas

    def _fetch_dynamical(self, model_name: str, init_time: str, forecast_hour: int):
        """Fetch vertical sounding from Dynamical.org for ECMWF/GFS."""
        if model_name.upper() not in SUPPORTED_MODELS:
            # Backward compatibility: non-supported Dynamical models still use Open-Meteo
            self._fetch_openmeteo(model_name)
            return

        ds = fetch_dynamical_forecast(
            model=model_name,
            lat=self.lat,
            lon=self.lon,
            init_time=init_time,
            forecast_hour=forecast_hour,
        )
        self.dataset = ds
        self.datos = self._dataset_to_legacy_wide(ds.df)
        self.source_name = f'dynamical:{model_name.upper()}'

    def _dataset_to_legacy_wide(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Convert LONG dataset to legacy wide columns for existing plotting."""
        long = df_long.copy()
        long['time'] = pd.to_datetime(long['time'])
        wide = pd.DataFrame({'time': sorted(long['time'].unique())})

        for level, chunk in long.groupby('pressure'):
            level_i = int(round(level))
            chunk = chunk.sort_values('time')
            merged = chunk[['time', 'temperature', 'relative_humidity', 'wind_speed',
                            'wind_direction', 'geopotential_height']].rename(columns={
                'temperature': f'temperature_{level_i}hPa',
                'relative_humidity': f'relative_humidity_{level_i}hPa',
                'wind_speed': f'wind_speed_{level_i}hPa',
                'wind_direction': f'wind_direction_{level_i}hPa',
                'geopotential_height': f'geopotential_height_{level_i}hPa',
            })
            wide = wide.merge(merged, on='time', how='left')

        wide['day_year'] = wide['time'].dt.dayofyear
        for level in sorted(long['pressure'].unique()):
            li = int(round(level))
            ws = wide.get(f'wind_speed_{li}hPa')
            wd = wide.get(f'wind_direction_{li}hPa')
            if ws is not None and wd is not None:
                ws_kt = ws * 0.539957
                wide[f'u_{li}'] = -ws_kt * np.sin(np.radians(wd))
                wide[f'v_{li}'] = -ws_kt * np.cos(np.radians(wd))

        if 'geopotential_height_1000hPa' in wide.columns:
            z1000 = pd.to_numeric(wide['geopotential_height_1000hPa'], errors='coerce')
            wide['boundary_layer_height'] = np.maximum(z1000 - self.elev, 10)
        else:
            wide['boundary_layer_height'] = np.nan
        return wide

    def _fetch_era5_year(self, year: int):
        """Fetch ERA5 pressure-level data for a year, shift to current year."""
        year_now, month_init, day_init = self.fechas[0].split('-')
        _, month_end, day_end = self.fechas[1].split('-')
        fechas_era5 = [
            f'{year}-{month_init}-{day_init}',
            f'{year}-{month_end}-{day_end}'
        ]
        modelo = self.weather_models['ERA5']
        data = fetch_vertical(self.lat, self.lon, self.elev,
                              self.tzinfo, fechas_era5, modelo,
                              self.LEVELS_ALL)
        if data is None:
            return
        self._process_response(data, str(year))
        delta = int(year_now) - year
        self.datos['time'] = self.datos['time'] + pd.DateOffset(years=delta)
        self._skewt_modelo = self.weather_models['ERA5']
        self._skewt_fechas = fechas_era5

    def _process_response(self, data: dict, source_name: str):
        """Transform raw API response into processed DataFrame."""
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df['day_year'] = df['time'].dt.dayofyear

        # Coerce all numeric columns
        for level in self.LEVELS_ALL:
            for var in ['temperature', 'relative_humidity', 'wind_speed',
                        'wind_direction', 'geopotential_height']:
                col = f'{var}_{level}hPa'
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Wind u,v components (knots) for barb display levels
        for level in self.LEVELS_DISPLAY:
            ws = df[f'wind_speed_{level}hPa']
            wd = df[f'wind_direction_{level}hPa']
            ws_kt = ws * 0.539957  # km/h → knots
            df[f'u_{level}'] = -ws_kt * np.sin(np.radians(wd))
            df[f'v_{level}'] = -ws_kt * np.cos(np.radians(wd))

        # Inversions between display levels
        for i in range(len(self.LEVELS_DISPLAY) - 1):
            lower = self.LEVELS_DISPLAY[i]
            upper = self.LEVELS_DISPLAY[i + 1]
            t_lo = df[f'temperature_{lower}hPa']
            t_hi = df[f'temperature_{upper}hPa']
            df[f'inversion_{lower}_{upper}'] = t_hi > t_lo

        # BLH: API value with Richardson fallback
        df['boundary_layer_height'] = pd.to_numeric(
            df.get('boundary_layer_height'), errors='coerce')
        blh_ri = self._compute_blh_richardson(df)
        df['blh_richardson'] = blh_ri

        mask_missing = df['boundary_layer_height'].isna()
        df.loc[mask_missing, 'boundary_layer_height'] = df.loc[mask_missing, 'blh_richardson']

        n_filled = mask_missing.sum()
        n_total = len(df)
        if n_filled > 0:
            print(f"  BLH: {n_total - n_filled}/{n_total} from API, "
                  f"{n_filled}/{n_total} estimated via Richardson (Ri=0.25)")

        self.datos = df
        self.dataset = self._to_vertical_dataset(df, self.LEVELS_ALL)
        self.source_name = source_name

    # ══════════════════════════════════════════════════════════════════════
    #  TRANSFORMS — BOUNDARY LAYER HEIGHT
    # ══════════════════════════════════════════════════════════════════════

    def _compute_blh_richardson(self, df):
        """
        Estimate BLH using Bulk Richardson Number across 8 pressure levels.

        Ri(z) = g·(θv(z) - θv_sfc)·(z - z_sfc) / [θv_sfc · |Δu|²]

        When Ri crosses 0.25, linear interpolation gives BLH (meters AGL).
        """
        g = 9.81
        Ri_crit = 0.25
        levels = self.LEVELS_ALL

        blh_values = np.full(len(df), np.nan)

        for idx in range(len(df)):
            # Surface reference (lowest level)
            T_sfc = df[f'temperature_{levels[0]}hPa'].iat[idx]
            rh_sfc = df[f'relative_humidity_{levels[0]}hPa'].iat[idx]
            z_sfc = df[f'geopotential_height_{levels[0]}hPa'].iat[idx]
            ws_sfc = df[f'wind_speed_{levels[0]}hPa'].iat[idx]
            wd_sfc = df[f'wind_direction_{levels[0]}hPa'].iat[idx]

            if any(np.isnan(x) for x in [T_sfc, rh_sfc, z_sfc, ws_sfc, wd_sfc]):
                continue

            theta_v_sfc = self._theta_v(T_sfc, rh_sfc, levels[0])
            u_sfc = -ws_sfc / 3.6 * np.sin(np.radians(wd_sfc))
            v_sfc = -ws_sfc / 3.6 * np.cos(np.radians(wd_sfc))

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
                du2 = max((u - u_sfc)**2 + (v - v_sfc)**2, 0.01)

                Ri = g * (theta_v - theta_v_sfc) * dz / (theta_v_sfc * du2)

                if Ri >= Ri_crit:
                    if Ri != Ri_prev:
                        frac = (Ri_crit - Ri_prev) / (Ri - Ri_prev)
                        blh_z = z_prev + frac * (z - z_prev) - z_sfc
                    else:
                        blh_z = dz
                    blh_values[idx] = max(blh_z, 10)
                    break

                Ri_prev = Ri
                z_prev = z

        return blh_values

    @staticmethod
    def _theta_v(T_celsius, rh_pct, p_hPa):
        """
        Virtual potential temperature θv (K).

        θ  = T·(1000/p)^0.286
        e_s = 6.112·exp(17.67·T / (T+243.5))     (Magnus, hPa)
        r  = 0.622·e / (p - e)                    (mixing ratio)
        θv = θ·(1 + 0.61·r)
        """
        T_K = T_celsius + 273.15
        theta = T_K * (1000.0 / p_hPa) ** 0.286
        e_s = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
        e = rh_pct / 100.0 * e_s
        r = 0.622 * e / max(p_hPa - e, 0.1)
        return theta * (1 + 0.61 * r)

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTTING
    # ══════════════════════════════════════════════════════════════════════

    def profileplot(self, fechas: list[str] = None) -> plt.Figure:
        """
        Standalone time-height cross section.

        Parameters
        ----------
        fechas : optional date subrange

        Returns
        -------
        matplotlib.Figure
        """
        if self.datos is None:
            raise ValueError("No vertical data. Call get_data() first.")

        fechas_local = fechas if fechas else self.fechas
        init_date = pd.to_datetime(fechas_local[0])
        end_date = pd.to_datetime(fechas_local[1])

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        # Need a datos_ref-like structure for night shading
        vd = self.datos.copy()
        vd['time_mpl'] = mdates.date2num(vd['time'])
        vd = vd[(vd.time >= init_date) & (vd.time <= end_date)].copy()

        self._draw_profile(ax, vd, init_date, end_date)

        ax.set_xlim(mdates.date2num(init_date), mdates.date2num(end_date))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

        fig.suptitle(
            f'Vertical Profile — {self.source_name} | {self.name}',
            fontsize=12, fontweight='bold')
        fig.tight_layout()
        return fig

    def plot_on_axes(self, ax, datos_ref, init_date, end_date):
        """
        Draw vertical profile on a given axes (for MeteoSfc.meteoplot integration).

        Parameters
        ----------
        ax : matplotlib.Axes — target axes
        datos_ref : DataFrame — surface reference data (needs time_mpl, is_day)
        init_date, end_date : datetime
        """
        if self.datos is None:
            ax.text(0.5, 0.5, 'No vertical data',
                    transform=ax.transAxes, ha='center', va='center')
            return

        vd = self.datos.copy()
        vd['time_mpl'] = mdates.date2num(vd['time'])
        vd = vd[(vd.time >= init_date) & (vd.time <= end_date)].copy()

        if vd.empty:
            ax.text(0.5, 0.5, 'No vertical data in range',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # Night shading from surface reference
        levels = self.LEVELS_DISPLAY
        ax.fill_between(
            datos_ref['time_mpl'], min(levels) - 50, max(levels) + 50,
            where=(datos_ref.is_day == 0).fillna(False),
            alpha=0.08, color='lightblue', zorder=0)

        self._draw_profile(ax, vd, init_date, end_date)
        ax.set_title(f'Vertical Profile — {self.source_name}',
                     fontsize=9, loc='left')

    def _draw_profile(self, ax, vd, init_date, end_date):
        """Core drawing logic for the time-height cross section."""
        if vd.empty:
            return

        levels = self.LEVELS_DISPLAY

        # ── Inversion shading ──
        layer_colors = {
            f'inversion_{levels[0]}_{levels[1]}': ('royalblue', 0.2),
            f'inversion_{levels[1]}_{levels[2]}': ('darkorange', 0.2),
            f'inversion_{levels[2]}_{levels[3]}': ('darkorange', 0.15),
        }
        for i in range(len(levels) - 1):
            lower, upper = levels[i], levels[i + 1]
            col_name = f'inversion_{lower}_{upper}'
            color, alpha = layer_colors[col_name]
            inv_mask = vd[col_name].fillna(False).values.astype(bool)
            ax.fill_between(vd['time_mpl'], lower, upper,
                            where=inv_mask, color=color, alpha=alpha, zorder=1,
                            label=f'Inv. {lower}-{upper}' if i == 0 else None)

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
        blh_m = pd.to_numeric(vd['boundary_layer_height'], errors='coerce').values
        blh_p = np.where(np.isfinite(blh_m),
                         1013.25 * np.exp(-blh_m / 8500.0), np.nan)
        ax.plot(vd['time_mpl'], blh_p,
                color='red', linewidth=2, linestyle='--',
                label='BLH', zorder=6)

        # ── Axes config ──
        ax.set_ylim(max(levels) + 20, min(levels) - 20)
        ax.set_yticks(levels)
        ax.set_yticklabels([str(l) for l in levels])
        ax.set_ylabel('Pressure (hPa)')
        ax.grid(True, alpha=0.3)

        # Right axis with approximate altitudes
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(levels)
        ax2.set_yticklabels([f'{self.ALTITUDES[l]} m' for l in levels])
        ax2.set_ylabel('Altitude (m ASL)')

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=7, ncol=2).set_zorder(10)

    # ── Skew-T ───────────────────────────────────────────────────────────

    def _fetch_skewt_data(self):
        """Fetch extended pressure levels (LEVELS_SKEWT) for Skew-T on demand."""
        if self._skewt_modelo is None:
            raise ValueError("Call get_data() first.")
        if self._skewt_modelo.get('keyword') == 'era5':
            ds = fetch_era5_arco(
                self.lat, self.lon,
                self._skewt_fechas[0], self._skewt_fechas[1],
                levels=self.LEVELS_SKEWT,
            )
            self._datos_skewt = self._dataset_to_legacy_wide(ds.df)
            return

        data = fetch_vertical(self.lat, self.lon, self.elev,
                              self.tzinfo, self._skewt_fechas, self._skewt_modelo,
                              self.LEVELS_SKEWT)
        if data is None:
            self._datos_skewt = self.datos
            return
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        for level in self.LEVELS_SKEWT:
            for var in ['temperature', 'relative_humidity', 'wind_speed', 'wind_direction', 'geopotential_height']:
                col = f'{var}_{level}hPa'
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        self._datos_skewt = df

    def _skewt_arrays(self, time: str):
        """
        Extract profile arrays from _datos_skewt for a given time.
        Returns (actual_time, pressure_vals, temp_vals, dewp_vals, u_vals, v_vals).
        """
        if self._datos_skewt is None:
            self._fetch_skewt_data()
        t = pd.to_datetime(time)
        idx = (self._datos_skewt['time'] - t).abs().idxmin()
        row = self._datos_skewt.loc[idx]
        actual_time = self._datos_skewt.loc[idx, 'time']

        pressure_vals, temp_vals, dewp_vals, u_vals, v_vals, z_vals = [], [], [], [], [], []
        for level in sorted(self.LEVELS_SKEWT, reverse=True):
            col_t  = f'temperature_{level}hPa'
            col_rh = f'relative_humidity_{level}hPa'
            col_ws = f'wind_speed_{level}hPa'
            col_wd = f'wind_direction_{level}hPa'
            T  = row[col_t]  if col_t  in row.index else np.nan
            rh = row[col_rh] if col_rh in row.index else np.nan
            if pd.isna(T) or pd.isna(rh):
                continue
            e_s = 6.112 * np.exp(17.67 * T / (T + 243.5))
            e   = (rh / 100.0) * e_s
            Td  = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
            ws  = row[col_ws] if col_ws in row.index else np.nan
            wd  = row[col_wd] if col_wd in row.index else np.nan
            ws_kt = ws * 0.539957 if not pd.isna(ws) else np.nan
            u = (-ws_kt * np.sin(np.radians(wd))
                 if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan)
            v = (-ws_kt * np.cos(np.radians(wd))
                 if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan)
            pressure_vals.append(level)
            temp_vals.append(T)
            dewp_vals.append(Td)
            u_vals.append(u)
            v_vals.append(v)
            z_vals.append(row.get(f'geopotential_height_{level}hPa', np.nan))
        return actual_time, pressure_vals, temp_vals, dewp_vals, u_vals, v_vals, z_vals

    def compute_skewt_indices(self, time: str) -> dict:
        """
        Compute thermodynamic indices for a single time step.

        Returns dict with keys: cape, cin, lcl_hpa, lcl_temp, trigger_temp.
        All temperatures in °C, pressures in hPa, energy in J/kg.
        Returns {} on any error.
        """
        try:
            import metpy.calc as mpcalc
            from metpy.units import units as munits
        except ImportError:
            return {}
        try:
            _, pv, tv, tdv, _, _, _ = self._skewt_arrays(time)
            if len(pv) < 3:
                return {}
            p     = np.array(pv) * munits.hPa
            T_arr = np.array(tv) * munits.degC
            Td_arr= np.array(tdv) * munits.degC

            # LCL
            p_lcl, T_lcl = mpcalc.lcl(p[0], T_arr[0], Td_arr[0])

            # CAPE / CIN
            parcel = mpcalc.parcel_profile(p, T_arr[0], Td_arr[0])
            cape, cin = mpcalc.cape_cin(p, T_arr, Td_arr, parcel)

            # Trigger temperature via CCL
            trigger_temp = None
            try:
                p_ccl, T_ccl = mpcalc.ccl(p, T_arr, Td_arr)[:2]
                T_trigger = mpcalc.dry_lapse(
                    np.atleast_1d(p[0]), T_ccl, reference_pressure=p_ccl
                )[0]
                trigger_temp = float(T_trigger.to('degC').magnitude)
            except Exception:
                pass

            return {
                'cape':         round(float(cape.magnitude), 1),
                'cin':          round(float(cin.magnitude), 1),
                'lcl_hpa':      round(float(p_lcl.magnitude), 1),
                'lcl_temp':     round(float(T_lcl.to('degC').magnitude), 1),
                'trigger_temp': round(trigger_temp, 1) if trigger_temp is not None else None,
            }
        except Exception:
            return {}

    def skewt(self, time: str) -> plt.Figure:
        """
        Plot Skew-T log-P diagram for a single time step.

        Parameters
        ----------
        time : str — 'YYYY-MM-DD HH:MM' timestamp (nearest match used)

        Returns
        -------
        matplotlib.Figure
        """
        try:
            from metpy.plots import SkewT
            from metpy.units import units as munits
        except ImportError:
            raise ImportError("MetPy is required for Skew-T: pip install metpy")

        if self.datos is None:
            raise ValueError("No vertical data. Call get_data() first.")

        actual_time, pressure_vals, temp_vals, dewp_vals, u_vals, v_vals, z_vals = \
            self._skewt_arrays(time)

        if not pressure_vals:
            raise ValueError("No valid data for the selected time step.")

        p = np.array(pressure_vals) * munits.hPa
        T_arr = np.array(temp_vals) * munits.degC
        Td_arr = np.array(dewp_vals) * munits.degC
        u_arr = np.array(u_vals) * munits.knots
        v_arr = np.array(v_vals) * munits.knots

        fig = plt.figure(figsize=(8, 9))
        skew = SkewT(fig, rotation=45)

        skew.plot(p, T_arr, 'r', linewidth=2, label='Temperature')
        skew.plot(p, Td_arr, 'g', linewidth=2, label='Dewpoint')

        valid_wind = np.isfinite(u_arr.magnitude) & np.isfinite(v_arr.magnitude)
        if valid_wind.any():
            skew.plot_barbs(p[valid_wind], u_arr[valid_wind], v_arr[valid_wind])

        skew.plot_dry_adiabats(alpha=0.25, linewidths=0.7)
        skew.plot_moist_adiabats(alpha=0.25, linewidths=0.7)
        skew.plot_mixing_lines(alpha=0.2, linewidths=0.7)

        skew.ax.set_ylim(1050, min(pressure_vals) - 10)
        skew.ax.set_xlim(-30, 45)
        skew.ax.set_xlabel('Temperature (°C)')
        skew.ax.set_ylabel('Pressure (hPa)')
        # Right axis: model geopotential height per pressure level
        if len(z_vals) == len(pressure_vals):
            ax2 = skew.ax.twinx()
            ax2.set_ylim(skew.ax.get_ylim())
            ax2.set_yticks(pressure_vals)
            ax2.set_yticklabels([
                f'{int(z)} m' if np.isfinite(z) else '—'
                for z in z_vals
            ])
            ax2.set_ylabel('Height (m ASL)')

        time_label = actual_time.strftime('%Y-%m-%d %H:%M UTC') \
            if hasattr(actual_time, 'strftime') else str(actual_time)
        skew.ax.set_title(
            f'Skew-T Log-P — {self.source_name} | {self.name}\n{time_label}',
            fontsize=10, fontweight='bold',
        )
        skew.ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        return fig
