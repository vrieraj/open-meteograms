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


class MeteoVrt:
    """Vertical profile data and visualization."""

    # Levels for data retrieval, BLH calculation, and Skew-T
    LEVELS_ALL = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500]

    # 4 levels for barb display in time-height cross section (less clutter)
    LEVELS_DISPLAY = [1000, 925, 850, 700]

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
        self.source_name = None
        self.weather_models = WEATHER_MODELS.copy()

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

        else:
            raise ValueError(f"Unknown source: '{source}'. "
                             f"Available: 'openmeteo', 'era5'")

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
        # Shift dates to current year
        delta = int(year_now) - year
        self.datos['time'] = self.datos['time'] + pd.DateOffset(years=delta)

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

        t = pd.to_datetime(time)
        idx = (self.datos['time'] - t).abs().idxmin()
        row = self.datos.loc[idx]
        actual_time = self.datos.loc[idx, 'time']

        pressure_vals, temp_vals, dewp_vals, u_vals, v_vals = [], [], [], [], []

        for level in sorted(self.LEVELS_ALL, reverse=True):  # surface → upper
            col_t = f'temperature_{level}hPa'
            col_rh = f'relative_humidity_{level}hPa'
            col_ws = f'wind_speed_{level}hPa'
            col_wd = f'wind_direction_{level}hPa'

            T = row[col_t] if col_t in row.index else np.nan
            rh = row[col_rh] if col_rh in row.index else np.nan
            if pd.isna(T) or pd.isna(rh):
                continue

            # Dewpoint via Magnus formula (°C)
            e_s = 6.112 * np.exp(17.67 * T / (T + 243.5))
            e = (rh / 100.0) * e_s
            Td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

            ws = row[col_ws] if col_ws in row.index else np.nan
            wd = row[col_wd] if col_wd in row.index else np.nan
            ws_kt = ws * 0.539957 if not pd.isna(ws) else np.nan
            u = -ws_kt * np.sin(np.radians(wd)) if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan
            v = -ws_kt * np.cos(np.radians(wd)) if not pd.isna(ws_kt) and not pd.isna(wd) else np.nan

            pressure_vals.append(level)
            temp_vals.append(T)
            dewp_vals.append(Td)
            u_vals.append(u)
            v_vals.append(v)

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

        time_label = actual_time.strftime('%Y-%m-%d %H:%M UTC') \
            if hasattr(actual_time, 'strftime') else str(actual_time)
        skew.ax.set_title(
            f'Skew-T Log-P — {self.source_name} | {self.name}\n{time_label}',
            fontsize=10, fontweight='bold',
        )
        skew.ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        return fig
