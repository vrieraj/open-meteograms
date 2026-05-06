# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meteogram generator that fetches hourly weather forecasts from the [Open-Meteo API](https://open-meteo.com/) and renders 4-panel wildfire weather charts. Primary use cases: multi-model NWP comparison, ERA5 historical overlay, boundary layer analysis, and observed station data overlay.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root for API keys (already in `.gitignore`):

```
WU_API_KEY=your_weather_underground_api_key
```

Run the Streamlit viewer:

```bash
streamlit run app.py
```

Run the Jupyter notebook (alternative workflow):

```bash
jupyter notebook example_use.ipynb
```

There is no test suite and no linter configured.

## Architecture

`OpenMeteograms.py` is a legacy monolithic file kept for reference. The canonical implementation lives in `scripts/` and `datasources/`:

| File | Responsibility |
|------|---------------|
| `app.py` | Streamlit viewer — map, search, sidebar controls, meteogram dialog |
| `scripts/place.py` | `nominatim()` geocoding + `Place` (lat/lon, elevation, timezone, utility URLs) |
| `scripts/meteo_sfc.py` | `MeteoSfc` — surface data pipeline and 4-panel meteogram |
| `scripts/meteo_vrt.py` | `MeteoVrt` — vertical pressure-level data, time-height cross section, BLH; `profileplot()` standalone, `plot_on_axes()` for meteoplot integration |
| `scripts/weather_models.py` | `WEATHER_MODELS` dict — NWP model definitions (keyword, type, resolution, etc.) |
| `datasources/openmeteo.py` | Thin HTTP layer: `fetch_surface()` and `fetch_vertical()` |
| `datasources/wx_stations.py` | WU PWS station discovery (`fetch_wu_stations_near`) and hourly history (`fetch_wu_hourly`) |
| `scripts/wx_scraper/wow_metie_scraper.py` | CLI: download WOW Met Éireann single-station observations to CSV |
| `scripts/wx_scraper/wow_stations_geojson.py` | CLI: discover active WOW Met Éireann stations → GeoJSON |

Imports within `scripts/` use relative imports (`from .weather_models import WEATHER_MODELS`). `datasources` is imported as an absolute package from the project root.

### Typical data flow (notebook / script)

```python
from scripts.place import nominatim, Place
from scripts.meteo_sfc import MeteoSfc
from scripts.meteo_vrt import MeteoVrt
from datasources.wx_stations import fetch_wu_hourly

# 1. Geocode
results = nominatim("Barcelona")
place = Place(results[0])

# 2. Create data objects
sfc = MeteoSfc(place, ['2025-07-01', '2025-07-07'])
vrt = MeteoVrt(place, ['2025-07-01', '2025-07-07'])   # optional

# 3. Fetch — calls accumulate into self.datos via pd.concat
sfc.get_data('openmeteo', models=['IFS', 'GFS', 'AROME'])
sfc.get_data('era5', years=[2023, 2024])
vrt.get_data('openmeteo', model='IFS')

# 4. Add station data (WU PWS)
df = fetch_wu_hourly('IMADR123', '2025-07-01', '2025-07-07', api_key)
sfc.get_data_station(df)   # source = stationId; renders as scatter markers

# 5. Plot
fig = sfc.meteoplot(vrt=vrt)   # returns matplotlib.Figure
```

### Key design patterns

- **Accumulator pattern (`MeteoSfc` only)**: `MeteoSfc.datos` is a DataFrame that grows with each `get_data()` / `get_data_station()` call via `pd.concat`. A `source` column identifies data origin (model name, year string like `'2024'`, or station ID).
- **`MeteoVrt` is single-source, not accumulated**: `MeteoVrt.datos` is replaced on each `get_data()` call. Only the most recent fetch is retained.
- **ERA5 year overlay**: Historical ERA5 data is date-shifted to the current year (`+ pd.DateOffset(years=delta)`) so that multiple years plot on the same x-axis.
- **BLH fallback**: `MeteoVrt` uses the API's `boundary_layer_height` when available; missing values are estimated via Bulk Richardson number (Ri_crit=0.25, 8 pressure levels from 1000–700 hPa).
- **Vertical vs. wind panel**: Panel 0 of `meteoplot()` shows a time-height cross section only when `vrt` is passed, there is exactly one source, that source is a forecast model (`type == 'forecast'`), and it is not a station. Otherwise it shows wind direction arrows. Archive-only or station-only runs always show wind direction arrows.
- **Station sources (`station_sources`)**: `MeteoSfc` keeps a `set[str]` of station IDs added via `get_data_station()`. `meteoplot()` renders these as scatter markers (`marker='o'`, `s=18`) instead of lines, using the same palette color as the source's position in `source_list`. Legends use a dot handle for stations.
- **Station data columns**: `fetch_wu_hourly()` returns a DataFrame with the same column names as Open-Meteo (`temperature_2m`, `dew_point_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_gusts_10m`, `wind_direction_10m`). `vapour_pressure_deficit` is NaN (WU does not provide it), so VPD-based FM is not computed for stations. Fosberg FM and wind arrows are computed normally.
- **`is_day` for stations**: derived from local hour (06–20 inclusive) since WU does not provide a solar flag.
- **`datos_ref`**: Always the first source in `source_list`. Drives night shading and the ignition semaphore regardless of how many sources are loaded.
- **`Place.__init__` makes HTTP calls**: Elevation is fetched from Open-Meteo and timezone is resolved via `timezonefinder` during construction — avoid instantiating `Place` in hot paths.
- **`app.py` event-queue pattern**: User interactions (map click, search selection) are stored in `st.session_state.event` and consumed before the map renders each run. `last_click` / `last_selected` deduplication prevents infinite rerun loops caused by `st_folium` and `st_searchbox` persisting their values across reruns. Both `nominatim_search()` (forward) and `nominatim_reverse()` (map click) are `@st.cache_data(ttl=86400)`. Map clicks use reverse geocoding to build a `Place`; search uses `nominatim_search` + `st.session_state.search_results` dict to look up the selected feature.
- **`app.py` uses `MeteoVrt` for single forecast-model runs only**: The meteogram dialog instantiates `MeteoVrt` only when exactly one model is selected and its `type == 'forecast'`. Archive models and station-only runs skip `MeteoVrt`.
- **Date range clamped to 10 days**: The sidebar validates the date picker and clamps `d_end = d_start + timedelta(days=10)` if the user selects a wider window. December→January year-crossing is the only supported cross-year range.
- **WU station discovery**: `fetch_wu_stations_near(lat, lon, radius_km, api_key)` converts the radius to an axis-aligned bbox, maps it to WU tile coordinates (lod=9, tile-size=512, effective zoom 8), and queries the tile API. Returns current observations alongside station metadata. Results are cached in `app.py` with `@st.cache_data(ttl=3600)`.
- **WU API key**: stored in `.env` (`WU_API_KEY=...`), loaded at startup via `python-dotenv`. Pre-fills the sidebar input; can be overridden interactively.
- **`skewt()` is unimplemented**: `MeteoVrt.skewt()` raises `NotImplementedError`; placeholder for future MetPy integration.

### Available weather models (`WEATHER_MODELS` keys)

| Key | Provider | Type | Resolution |
|-----|----------|------|-----------|
| `ICON` | DWD (Germany) | forecast | 2–11 km |
| `GFS` | NOAA (USA) | forecast | 3–25 km |
| `AROME` | MeteoFrance | forecast | 1–25 km |
| `IFS` | ECMWF | forecast | 25 km |
| `GSM JMA` | JMA (Japan) | forecast | 5–55 km |
| `MET Nordic` | MET Norway | forecast | 1 km |
| `GEM` | Canadian Weather Service | forecast | 2.5 km |
| `GFS GRAPES` | CMA (China) | forecast | 15 km |
| `ACCESS-G` | BOM (Australia) | forecast | 15 km |
| `COSMO` | AM ARPAE ARPAP (Italy) | forecast | 2 km |
| `UKMO` | UK Met Office | forecast | 2–10 km |
| `KNMI` | KNMI (Netherlands) | forecast | 2 km |
| `DMI` | DMI (Denmark) | forecast | 2 km |
| `ECMWF IFS` | ECMWF | archive | 9 km |
| `ERA5` | ECMWF | archive | 11 km |

### Open-Meteo API endpoints

- Forecast models: `https://api.open-meteo.com/v1/forecast`
- Archive models (ERA5, ECMWF IFS): `https://archive-api.open-meteo.com/v1/archive`
- The `type` field in `WEATHER_MODELS` (`'forecast'` vs `'archive'`) controls which endpoint is used.

### WU PWS API endpoints

- Station discovery (tiles): `https://api.weather.com/v2/vector-api/products/614/features`
- Hourly history: `https://api.weather.com/v2/pws/history/hourly`
- Requires a Weather Underground API key (`WU_API_KEY` in `.env`).

### Meteogram panels

| Panel | Content |
|-------|---------|
| TOP | Vertical profile (BLH + barbs + inversions) or wind direction arrows |
| WIND | Wind speed + gusts (km/h) |
| TEMP | Temperature + dew point (°C, left axis) + relative humidity (%, right axis) |
| FUEL | 1-h fuel moisture: Fosberg table and VPD-based (Resco de Dios) + ignition probability semaphore |

### Fire weather variables

- **Fosberg FM**: lookup table (Fosberg Table A, NFDRS) indexed by `is_day`, temperature bin, and RH.
- **VPD FM** (Resco de Dios 2015/2024): `FM = 3.5 + 28 * exp(-1.5 * VPD_kPa)`. Not computed for station sources (no VPD available).
- **Ignition probability**: 9×16 lookup table indexed by temperature (5°C bins) × fuel moisture.
- The ignition semaphore is a colored bar at the bottom of the fuel panel using the first source in `datos_ref`.
