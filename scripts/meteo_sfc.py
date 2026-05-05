"""
Surface meteorology: data pipeline and meteogram plotting.

Pipeline: get_data(source, **kwargs) → normalize → self.datos (DataFrame)
Transforms: Fosberg FM, VPD FM (Resco de Dios), probability of ignition.
Plotting: meteoplot() — 4-panel wildfire weather meteogram.

Column conventions:
    source   — data origin (e.g. 'AROME', '2024', 'METAR:LECO')
    day_year — day of year (1–366), useful for inter-annual comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.lines as mlines

from .weather_models import WEATHER_MODELS
from datasources.openmeteo import fetch_surface


class MeteoSfc:
    """Surface meteorological data and wildfire weather meteogram."""

    def __init__(self, place, fechas: list[str]):
        """
        Parameters
        ----------
        place : Place — geographic location
        fechas : list — ['YYYY-MM-DD', 'YYYY-MM-DD'] date range
        """
        props  = getattr(place, 'properties', {}) or {}
        county = props.get('county') or ''
        state  = props.get('state') or ''
        inner  = ', '.join(filter(None, [county, state]))
        self.name = f"{place.name} ({inner})" if inner else place.name
        self.lat = place.lat
        self.lon = place.lon
        self.elev = place.elev
        self.tzinfo = place.tzinfo
        self.fechas = fechas
        self.datos = pd.DataFrame()
        self.weather_models = WEATHER_MODELS.copy()

    # ══════════════════════════════════════════════════════════════════════
    #  DATA PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    def get_data(self, source: str = 'openmeteo', **kwargs) -> pd.DataFrame:
        """
        Unified data pipeline. Downloads, transforms and appends to self.datos.

        Parameters
        ----------
        source : str
            'openmeteo' — NWP forecast models (requires models=[...])
            'era5'      — ERA5 reanalysis by year (requires years=[...])
            Future: 'metar', 'stations', 'wunderground', ...

        Returns
        -------
        pd.DataFrame — accumulated self.datos
        """
        if source == 'openmeteo':
            models = kwargs.get('models', [])
            self._fetch_openmeteo_models(models)

        elif source == 'era5':
            years = kwargs.get('years', [])
            self._fetch_openmeteo_years(years)

        else:
            raise ValueError(f"Unknown source: '{source}'. "
                             f"Available: 'openmeteo', 'era5'")

        return self.datos

    def _fetch_openmeteo_models(self, models: list):
        """Fetch forecast models from Open-Meteo."""
        for model_name in models:
            if model_name not in self.weather_models:
                print(f"  Warning: unknown model '{model_name}', skipping")
                continue
            modelo = self.weather_models[model_name]
            data = fetch_surface(self.lat, self.lon, self.elev,
                                 self.tzinfo, self.fechas, modelo)
            if data is None:
                continue
            df = self._transform(pd.DataFrame(data['hourly']))
            df['source'] = model_name
            self.datos = pd.concat([self.datos, df], ignore_index=True)

    def _fetch_openmeteo_years(self, years: list):
        """Fetch ERA5 reanalysis for specific years, shift dates to current year."""
        year_now, month_init, day_init = self.fechas[0].split('-')
        _, month_end, day_end = self.fechas[1].split('-')

        modelo = self.weather_models['ERA5']
        for year in years:
            fechas_era5 = [
                f'{year}-{month_init}-{day_init}',
                f'{year}-{month_end}-{day_end}'
            ]
            data = fetch_surface(self.lat, self.lon, self.elev,
                                 self.tzinfo, fechas_era5, modelo)
            if data is None:
                continue
            df = self._transform(pd.DataFrame(data['hourly']))
            df['source'] = str(year)
            # Shift dates to current year for overlay plotting
            delta = int(year_now) - year
            df['time'] = df['time'] + pd.DateOffset(years=delta)
            self.datos = pd.concat([self.datos, df], ignore_index=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TRANSFORMS
    # ══════════════════════════════════════════════════════════════════════

    def _transform(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Normalize and compute derived variables."""
        datos = datos.copy(deep=True)
        datos['time'] = pd.to_datetime(datos['time'])
        datos['day_year'] = datos['time'].dt.dayofyear
        datos['wind_direction_arrow'] = self._wind_arrows(datos)
        datos['fuel_moisture'] = self._fuel_moisture_fosberg(datos)
        datos['fuel_moisture_vpd'] = self._fuel_moisture_vpd(datos)
        datos['prob_ignition'] = self._prob_ignition(datos)
        return datos

    @staticmethod
    def _wind_arrows(datos):
        return pd.cut(
            datos['wind_direction_10m'] % 360,
            bins=[-1, 23, 67, 112, 157, 202, 247, 292, 337, 361],
            labels=[
                r'$\downarrow$', r'$\swarrow$', r'$\leftarrow$',
                r'$\nwarrow$', r'$\uparrow$', r'$\nearrow$',
                r'$\rightarrow$', r'$\searrow$', r'$\downarrow$'
            ],
            right=False, ordered=False
        )

    @staticmethod
    def _fuel_moisture_fosberg(datos):
        """Fosberg Table A — classic NFDRS 1-h fuel moisture (SIN corregir)."""
        tabla = {
            'dia': {
                't10': [1,2,2,3,4,5,5,6,7,7,7,8,9,9,10,10,11,12,13,13,13],
                't21': [1,2,2,3,4,5,5,6,6,7,7,8,8,9,9,10,11,12,12,12,13],
                't32': [1,1,2,2,3,4,5,5,6,7,7,8,8,8,9,10,10,11,12,12,13],
                't43': [1,1,2,2,3,4,4,5,6,7,7,8,8,8,9,10,10,11,12,12,13],
                'tmax':[1,1,2,2,3,4,4,5,6,7,7,8,8,8,9,10,10,11,12,12,13]
            },
            'noche': {
                't10': [1,2,3,4,5,6,7,8,9,9,11,11,12,13,14,16,18,21,24,25,25],
                't21': [1,2,3,4,5,6,6,8,8,9,10,11,11,12,14,16,17,20,23,25,25],
                't32': [1,2,3,4,4,5,6,7,8,9,10,10,11,12,13,15,17,20,23,25,25],
                't43': [1,2,3,3,4,5,6,7,8,9,9,10,10,11,13,14,16,19,22,25,25],
                'tmax':[1,2,2,3,4,5,6,6,8,8,9,9,10,11,12,14,16,19,21,24,25]
            }
        }

        def get_temp_key(temp):
            if temp is None or np.isnan(temp):
                return None
            elif temp < 10:  return 't10'
            elif temp < 21:  return 't21'
            elif temp < 32:  return 't32'
            elif temp < 43:  return 't43'
            else:            return 'tmax'

        values = []
        for _, row in datos.iterrows():
            period = 'dia' if row['is_day'] == 1 else 'noche'
            tkey = get_temp_key(row['temperature_2m'])
            if tkey is None or pd.isna(row['relative_humidity_2m']):
                values.append(None)
                continue
            hum_idx = int(np.clip(round(row['relative_humidity_2m'] / 5), 0, 20))
            values.append(tabla[period][tkey][hum_idx])
        return values

    @staticmethod
    def _fuel_moisture_vpd(datos):
        """Resco de Dios et al. (2015, 2024) — VPD-based model.
        FM = 3.5 + 28.0 * exp(-1.5 * VPD_kPa)
        Calibrated on BONFIRE global dataset (1603 records)."""
        vpd = datos.get('vapour_pressure_deficit')
        if vpd is None:
            return [None] * len(datos)
        FM0, FM1, m = 3.5, 28.0, 1.5
        fm = FM0 + FM1 * np.exp(-m * vpd)
        return fm.where(vpd.notna(), other=None).tolist()

    @staticmethod
    def _prob_ignition(datos):
        """Probability of ignition lookup (temperature × fuel moisture)."""
        tabla = [
            [90,70,60,60,50,40,40,30,30,20,20,20,10,10,10,10],
            [90,70,60,60,50,40,40,30,30,20,20,20,10,10,10,10],
            [90,80,70,60,50,40,40,30,30,20,20,20,10,10,10,10],
            [90,80,70,60,50,40,40,30,30,20,20,20,10,10,10,10],
            [100,80,70,60,60,50,40,40,30,30,20,20,20,10,10,10],
            [100,90,80,70,60,50,40,40,30,30,20,20,20,20,10,10],
            [100,90,80,70,60,50,50,40,30,30,30,20,20,20,10,10],
            [100,90,80,70,60,60,50,40,40,30,30,20,20,20,10,10],
            [100,100,90,80,70,60,50,40,40,30,30,30,20,20,20,10]
        ]
        values = []
        for _, row in datos.iterrows():
            t = row['temperature_2m']
            fm = row['fuel_moisture']
            if pd.isna(t) or fm is None or pd.isna(fm):
                values.append(None)
                continue
            t_idx = int(np.clip(t / 5, 0, 8))
            h_idx = int(np.clip(fm - 2, 0, 15))
            values.append(tabla[t_idx][h_idx])
        return values

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTTING — METEOGRAM
    # ══════════════════════════════════════════════════════════════════════

    def meteoplot(self, fechas: list[str] = [], sources: list[str] = [],
                  vrt=None) -> plt.Figure:
        """
        4-panel wildfire weather meteogram.

        Parameters
        ----------
        fechas : list — optional date subrange ['YYYY-MM-DD', 'YYYY-MM-DD']
        sources : list — filter by source names (default: all)
        vrt : MeteoVrt — optional vertical profile object for panel 0

        Returns
        -------
        matplotlib.Figure
        """

        def get_partial_palette(cmap_name, start, stop, n_colors):
            cmap = plt.get_cmap(cmap_name)
            return [cmap(x) for x in np.linspace(start, stop, max(n_colors, 1))]

        # ── FILTER ────────────────────────────────────────
        fechas_local = fechas if fechas else self.fechas
        init_date = pd.to_datetime(fechas_local[0])
        end_date = pd.to_datetime(fechas_local[1])

        datos = self.datos.loc[
            (self.datos.time >= init_date) & (self.datos.time <= end_date)
        ]
        source_list = sources if sources else datos.source.unique()
        datos = datos[datos['source'].isin(source_list)]

        if datos.empty:
            raise ValueError(f'No data for range: {init_date.date()} — {end_date.date()}')

        invalid = set(source_list) - set(datos.source.unique())
        if invalid:
            raise ValueError(f'Unknown sources: {invalid}')

        source_list = datos.source.unique()

        # ── PREPARE ───────────────────────────────────────
        datos_plot = datos.copy()
        datos_plot['time_mpl'] = mdates.date2num(datos_plot['time'])
        datos_ref = datos_plot.loc[datos_plot.source == source_list[0]].copy()

        init_date_mpl = mdates.date2num(init_date)
        end_date_mpl = mdates.date2num(end_date)

        n_sources = len(source_list)
        single_source = (n_sources == 1)
        has_vertical = (vrt is not None and vrt.datos is not None and single_source)

        # ── FIGURE ────────────────────────────────────────
        total_rows = 4
        height_ratios = [1.4, 1, 1, 1] if has_vertical else [0.6, 1, 1, 1]

        fig, ax = plt.subplots(
            total_rows, 1, figsize=(10, 10),
            gridspec_kw={'height_ratios': height_ratios}
        )
        TOP, WIND, TEMP, FUEL = 0, 1, 2, 3
        ax_rh = ax[TEMP].twinx()
        ax[TEMP].set_zorder(ax_rh.get_zorder() + 1)
        ax[TEMP].patch.set_visible(False)

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
            a.fill_between(datos_ref['time_mpl'], 0, 100,
                           where=mask, alpha=0.1, color='lightblue', zorder=0)

        # ── PALETTES ──────────────────────────────────────
        colors = {
            'greens':  get_partial_palette('Greens_r',  0.1, 0.7, n_sources),
            'yellows': get_partial_palette('Wistia_r',  0.1, 0.7, n_sources),
            'reds':    get_partial_palette('Reds_r',    0.1, 0.7, n_sources),
            'blues':   get_partial_palette('Blues_r',   0.1, 0.7, n_sources),
            'greys':   get_partial_palette('Greys_r',   0.1, 0.7, n_sources),
        }

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

        def plot_var(feat, axis, data):
            cfg = VAR_CFG[feat]
            if single_source:
                d = data[data.source == source_list[0]]
                axis.plot(d['time_mpl'], d[feat],
                          color=cfg['color'], linestyle=cfg['style'],
                          label=cfg['label'], zorder=5)
            else:
                for idx, src in enumerate(source_list):
                    d = data[data.source == src]
                    axis.plot(d['time_mpl'], d[feat],
                              color=cfg['palette'][idx], linestyle=cfg['style'],
                              label=src, zorder=5)

        # ── PANEL 0: VERTICAL PROFILE or WIND DIRECTION ──
        if has_vertical:
            vrt.plot_on_axes(ax[TOP], datos_ref, init_date, end_date)
        else:
            self._plot_wind_direction(ax[TOP], datos_plot, source_list, datos_ref)

        # ── PANEL 1: WIND ─────────────────────────────────
        plot_var('wind_speed_10m', ax[WIND], datos_plot)
        plot_var('wind_gusts_10m', ax[WIND], datos_plot)
        ax[WIND].set_ylabel('Wind (km/h)')
        ax[WIND].grid(True, alpha=0.3)

        max_gusts = datos_plot['wind_gusts_10m'].max()
        if max_gusts <= 50:
            ax[WIND].set_ylim(0, 60)
        else:
            ax[WIND].set_ylim(0, None)

        if single_source:
            ax[WIND].legend(loc='upper left', fontsize=7)
        else:
            model_h = [mlines.Line2D([], [], color=colors['greens'][i], label=m)
                       for i, m in enumerate(source_list)]
            style_h = [mlines.Line2D([], [], color='grey', linestyle='-', label='Speed'),
                       mlines.Line2D([], [], color='grey', linestyle='--', label='Gusts')]
            ax[WIND].legend(handles=model_h + style_h,
                            loc='upper left', fontsize=7, ncol=2)

        # ── PANEL 2: TEMP + DEWPOINT + RH ─────────────────
        plot_var('temperature_2m', ax[TEMP], datos_plot)
        plot_var('dew_point_2m', ax[TEMP], datos_plot)
        plot_var('relative_humidity_2m', ax_rh, datos_plot)

        ax[TEMP].grid(True, alpha=0.3)
        ax[TEMP].set_ylabel('Temperature / Dewpoint (°C)')
        ax_rh.set_ylabel('Relative Humidity (%)')
        ax_rh.set_ylim(0, 100)
        ax_rh.set_yticks(range(0, 100, 10))

        t_max = datos_plot['temperature_2m'].max()
        t_min = datos_plot[['temperature_2m', 'dew_point_2m']].min().min()
        if t_min >= -5 and t_max <= 40:
            ax[TEMP].set_ylim(-5, 40)
            ax[TEMP].set_yticks(range(-5, 45, 5))

        if single_source:
            h1, l1 = ax[TEMP].get_legend_handles_labels()
            ax[TEMP].legend(h1, l1, loc='upper left', fontsize=7)
            h2, l2 = ax_rh.get_legend_handles_labels()
            ax_rh.legend(h2, l2, loc='upper right', fontsize=7)
        else:
            model_ht = [mlines.Line2D([], [], color=colors['reds'][i], label=m)
                        for i, m in enumerate(source_list)]
            style_ht = [mlines.Line2D([], [], color='grey', linestyle='-', label='Temp'),
                        mlines.Line2D([], [], color='grey', linestyle='--', label='Dewpoint')]
            ax[TEMP].legend(handles=model_ht + style_ht,
                            loc='upper left', fontsize=7, ncol=2)
            model_hr = [mlines.Line2D([], [], color=colors['blues'][i], label=m)
                        for i, m in enumerate(source_list)]
            ax_rh.legend(handles=model_hr, loc='upper right', fontsize=7,
                         title='RH %', title_fontsize=7)

        # ── PANEL 3: FUEL MOISTURE ────────────────────────
        for feat in ['fuel_moisture', 'fuel_moisture_vpd']:
            if datos_plot[feat].dropna().empty:
                continue
            plot_var(feat, ax[FUEL], datos_plot)

        ax[FUEL].set_ylabel('Fuel Moisture (%)')
        ax[FUEL].set_ylim(0, 35)
        ax[FUEL].set_yticks(range(0, 35, 5))
        ax[FUEL].grid(True, alpha=0.3)

        if single_source:
            ax[FUEL].legend(loc='upper left', fontsize=7)
        else:
            model_hf = [mlines.Line2D([], [], color=colors['yellows'][i], label=m)
                        for i, m in enumerate(source_list)]
            style_hf = [mlines.Line2D([], [], color='grey', linestyle='--', label='Fosberg'),
                        mlines.Line2D([], [], color='grey', linestyle='-', label='VPD')]
            ax[FUEL].legend(handles=model_hf + style_hf,
                            loc='upper left', fontsize=7, ncol=2)

        self._plot_ignition_semaphore(ax[FUEL], datos_ref)

        # ── TITLE ─────────────────────────────────────────
        fig.suptitle(
            f'Meteogram — {self.name}  |  {self.lat:.4f}°  {self.lon:.4f}°',
            fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12)
        return fig

    # ── Subplot helpers ───────────────────────────────────────────────────

    def _plot_wind_direction(self, ax, datos, source_list, datos_ref):
        ax.fill_between(datos_ref['time_mpl'], 0, 100,
                        where=(datos_ref.is_day == 0).fillna(False),
                        alpha=0.2, color='lightblue', zorder=0)
        for index, src in enumerate(source_list):
            datos_src = datos.loc[datos.source == src]
            for _, row in datos_src.iterrows():
                ax.text(mdates.date2num(row['time']), index + 1,
                        row['wind_direction_arrow'],
                        fontsize=18, ha='center', va='center', color='k', zorder=5)
        ax.set_ylim(0, len(source_list) + 1)
        ax.set_yticks(range(1, len(source_list) + 1))
        ax.set_yticklabels(source_list)

    @staticmethod
    def _plot_ignition_semaphore(ax_fuel, datos_ref):
        prob = datos_ref['prob_ignition'].values
        times = mdates.date2num(pd.to_datetime(datos_ref['time']))

        if all(p is None for p in prob):
            return

        ylim = ax_fuel.get_ylim()
        bar_height = (ylim[1] - ylim[0]) * 0.06
        bar_bottom = ylim[0]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'ignition', ['#2d9e2d', '#f0c929', '#e88a1a', '#d92525'], N=256)
        norm = mcolors.Normalize(vmin=0, vmax=100)

        for i in range(len(times) - 1):
            p = prob[i]
            if p is None or np.isnan(p):
                continue
            ax_fuel.axvspan(times[i], times[i + 1],
                            ymin=0, ymax=bar_height / (ylim[1] - ylim[0]),
                            color=cmap(norm(p)), alpha=0.8, zorder=4)

        ax_fuel.set_ylim(bar_bottom - bar_height * 0.5, ylim[1])
        ax_fuel.text(times[0], bar_bottom + bar_height * 0.3,
                     'Ignition Prob.', fontsize=7, va='center', ha='left',
                     color='white', fontweight='bold', zorder=5)
