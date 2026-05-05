# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meteogram generator that fetches hourly weather forecasts from the [Open-Meteo API](https://open-meteo.com/) and renders 4-panel wildfire weather charts. Primary use cases: multi-model NWP comparison, ERA5 historical overlay, and boundary layer analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
| `scripts/meteo_vrt.py` | `MeteoVrt` — vertical pressure-level data, time-height cross section, BLH |
| `scripts/weather_models.py` | `WEATHER_MODELS` dict — NWP model definitions (keyword, type, resolution, etc.) |
| `datasources/openmeteo.py` | Thin HTTP layer: `fetch_surface()` and `fetch_vertical()` |

Imports within `scripts/` use relative imports (`from .weather_models import WEATHER_MODELS`). `datasources` is imported as an absolute package from the project root.

### Typical data flow (notebook / script)

```python
from scripts.place import nominatim, Place
from scripts.meteo_sfc import MeteoSfc
from scripts.meteo_vrt import MeteoVrt

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

# 4. Plot
fig = sfc.meteoplot(vrt=vrt)   # returns matplotlib.Figure
```

### Key design patterns

- **Accumulator pattern (`MeteoSfc` only)**: `MeteoSfc.datos` is a DataFrame that grows with each `get_data()` call via `pd.concat`. A `source` column identifies data origin (model name or year string like `'2024'`).
- **`MeteoVrt` is single-source, not accumulated**: `MeteoVrt.datos` is replaced on each `get_data()` call. Only the most recent fetch is retained.
- **ERA5 year overlay**: Historical ERA5 data is date-shifted to the current year (`+ pd.DateOffset(years=delta)`) so that multiple years plot on the same x-axis.
- **BLH fallback**: `MeteoVrt` uses the API's `boundary_layer_height` when available; missing values are estimated via Bulk Richardson number (Ri_crit=0.25, 8 pressure levels from 1000–700 hPa).
- **Vertical vs. wind panel**: Panel 0 of `meteoplot()` shows a time-height cross section only when `vrt` is passed **and exactly one source** is in `datos`; otherwise it shows wind direction arrows for multi-model comparison.
- **`datos_ref`**: Always the first source in `source_list`. Drives night shading and the ignition semaphore regardless of how many sources are loaded.
- **`Place.__init__` makes HTTP calls**: Elevation is fetched from Open-Meteo and timezone is resolved via `timezonefinder` during construction — avoid instantiating `Place` in hot paths.
- **`app.py` event-queue pattern**: User interactions (map click, search selection) are stored in `st.session_state.event` and consumed before the map renders each run. `last_click` / `last_selected` deduplication prevents infinite rerun loops caused by `st_folium` and `st_searchbox` persisting their values across reruns. Nominatim calls are `@st.cache_data(ttl=86400)`.
- **`app.py` has no vertical data**: The Streamlit viewer only uses `MeteoSfc`; `MeteoVrt` is not wired in.
- **`skewt()` is unimplemented**: `MeteoVrt.skewt()` raises `NotImplementedError`; placeholder for future MetPy integration.

### Open-Meteo API endpoints

- Forecast models: `https://api.open-meteo.com/v1/forecast`
- Archive models (ERA5, ECMWF IFS): `https://archive-api.open-meteo.com/v1/archive`
- The `type` field in `WEATHER_MODELS` (`'forecast'` vs `'archive'`) controls which endpoint is used.

### Meteogram panels

| Panel | Content |
|-------|---------|
| TOP | Vertical profile (BLH + barbs + inversions) or wind direction arrows |
| WIND | Wind speed + gusts (km/h) |
| TEMP | Temperature + dew point (°C, left axis) + relative humidity (%, right axis) |
| FUEL | 1-h fuel moisture: Fosberg table and VPD-based (Resco de Dios) + ignition probability semaphore |

### Fire weather variables

- **Fosberg FM**: lookup table (Fosberg Table A, NFDRS) indexed by `is_day`, temperature bin, and RH.
- **VPD FM** (Resco de Dios 2015/2024): `FM = 3.5 + 28 * exp(-1.5 * VPD_kPa)`.
- **Ignition probability**: 9×16 lookup table indexed by temperature (5°C bins) × fuel moisture.
- The ignition semaphore is a colored bar at the bottom of the fuel panel using the first source in `datos_ref`.
