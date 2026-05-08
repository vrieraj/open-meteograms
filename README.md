# 🌤️ Open Meteograms — Wildfire Weather Viewer

A web-based meteogram generator focused on **wildfire weather** variables. Fetches hourly forecasts and historical reanalysis from the [Open-Meteo API](https://open-meteo.com/) and renders 4-panel fire-weather charts for any location worldwide.

Deployable as a standalone web app (Flask + Leaflet) or usable as a Python library via notebooks/scripts.

## Fire-weather panels

| Panel | Variables |
|-------|-----------|
| **TOP** | Boundary layer height · wind barbs · thermal inversions (vertical profile, single forecast model) or wind direction arrows (multi-model / archive) |
| **WIND** | Wind speed · gusts (km/h) |
| **TEMP** | Temperature · dew point (°C) · relative humidity (%) |
| **FUEL** | 1-h fuel moisture — Fosberg (NFDRS lookup table) and VPD-based (Resco de Dios 2015/2024) — plus ignition probability semaphore |

## Features

- **Multi-model NWP comparison** — ECMWF IFS, GFS, ICON, AROME, UKMO, and 10+ more
- **ERA5 historical overlay** — load multiple past years on the same x-axis for climatological context
- **Weather Underground PWS stations** — discover and overlay observed station data as scatter markers
- **Interactive Leaflet map** — click to select location; CartoDB / Esri satellite / hybrid basemaps
- **GeoJSON vector layers** — load fire perimeters or any polygon layer; colour picker per layer; auto-zoom to extent
- **Excel export** — download the full hourly dataset (one sheet per source)
- **10-day limit** with automatic date clamping

## Architecture

```
open-meteograms/
├── api/                    ← Flask API (entry point: api/app.py)
│   └── routes/
│       ├── search.py       ← /api/search  /api/reverse  /api/place
│       ├── meteogram.py    ← POST /api/meteogram → PNG
│       │                      POST /api/excel    → .xlsx
│       └── stations.py     ← /api/stations (WU PWS)
├── static/                 ← Frontend (Leaflet + vanilla JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── scripts/                ← Python core (no web dependencies)
│   ├── meteo_sfc.py        ← MeteoSfc — surface data pipeline + 4-panel meteogram
│   ├── meteo_vrt.py        ← MeteoVrt — vertical profiles + BLH
│   ├── place.py            ← Nominatim geocoding + Place object
│   └── weather_models.py   ← WEATHER_MODELS registry
├── datasources/
│   ├── openmeteo.py        ← Open-Meteo HTTP layer
│   └── wx_stations.py      ← Weather Underground PWS API
├── Dockerfile              ← gunicorn on port 7860 (Hugging Face Spaces)
└── requirements.txt
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: Weather Underground API key
echo "WU_API_KEY=your_key" > .env

python api/app.py
# → http://localhost:7860
```

## Deploy on Hugging Face Spaces

1. Create a Space with **SDK: Docker**
2. Add `WU_API_KEY` under **Settings → Secrets**
3. Push this repository to the Space:

```bash
git remote add hf https://huggingface.co/spaces/<user>/<space-name>
git push hf main
```

Hugging Face runs `docker build` automatically and exposes the app at `https://<user>-<space-name>.hf.space`.

## Python library usage

```python
from scripts.place import nominatim, Place
from scripts.meteo_sfc import MeteoSfc
from scripts.meteo_vrt import MeteoVrt

place = Place(nominatim("Newry, Northern Ireland")[0])
sfc   = MeteoSfc(place, ['2026-05-01', '2026-05-07'])
sfc.get_data('openmeteo', models=['IFS', 'GFS', 'UKMO'])
sfc.get_data('era5', years=[2023, 2024])

vrt = MeteoVrt(place, ['2026-05-01', '2026-05-07'])
vrt.get_data('openmeteo', model='IFS')

fig = sfc.meteoplot(vrt=vrt)
fig.savefig('meteogram.png', dpi=150, bbox_inches='tight')
```

## Data sources

- **Weather forecasts & reanalysis** — [Open-Meteo](https://open-meteo.com/) (free for non-commercial use)
- **Weather station observations** — [Weather Underground PWS API](https://www.wunderground.com/member/api-keys) (API key required)
- **Geocoding** — [Nominatim / OpenStreetMap](https://nominatim.org/)

## Attribution

Zippenfenig, P. *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7970649

Open-Meteo data is made available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence.

## Licence

This software is distributed under the **GNU General Public Licence v3.0** — see [LICENSE](LICENSE) for details.
