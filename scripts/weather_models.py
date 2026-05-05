"""
Weather model definitions for Open-Meteo API.

Each model entry contains:
    - provider:    National weather service or institution
    - keyword:     Open-Meteo API model identifier
    - country:     Country of origin
    - resolution:  Spatial resolution
    - days:        Forecast length or archive range
    - frequency:   Update frequency
    - type:        'forecast' or 'archive' (determines API endpoint)

Reference: https://open-meteo.com/en/docs
"""

WEATHER_MODELS = {
    "ICON": {
        "provider": "Deutscher Wetterdienst (DWD)",
        "keyword": "icon_seamless",
        "country": "Germany",
        "resolution": "2 - 11 km",
        "days": "7.5 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "GFS": {
        "provider": "NOAA",
        "keyword": "gfs_seamless",
        "country": "United States",
        "resolution": "3 - 25 km",
        "days": "16 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "AROME": {
        "provider": "MeteoFrance",
        "keyword": "meteofrance_seamless",
        "country": "France",
        "resolution": "1 - 25 km",
        "days": "4 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "IFS": {
        "provider": "ECMWF",
        "keyword": "ecmwf_ifs025",
        "country": "European Union",
        "resolution": "25 km",
        "days": "7 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "GSM JMA": {
        "provider": "JMA",
        "keyword": "jma_seamless",
        "country": "Japan",
        "resolution": "5 - 55 km",
        "days": "11 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "MET Nordic": {
        "provider": "MET Norway",
        "keyword": "metno_nordic",
        "country": "Norway",
        "resolution": "1 km",
        "days": "2.5 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "GEM": {
        "provider": "Canadian Weather Service",
        "keyword": "gem_seamless",
        "country": "Canada",
        "resolution": "2.5 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "GFS GRAPES": {
        "provider": "China Meteorological Administration (CMA)",
        "keyword": "cma_grapes_global",
        "country": "China",
        "resolution": "15 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "ACCESS-G": {
        "provider": "Australian Bureau of Meteorology (BOM)",
        "keyword": "bom_access_global",
        "country": "Australia",
        "resolution": "15 km",
        "days": "10 days",
        "frequency": "Every 6 hours",
        "type": "forecast"
    },
    "COSMO": {
        "provider": "AM ARPAE ARPAP",
        "keyword": "arpae_cosmo_seamless",
        "country": "Italy",
        "resolution": "2 km",
        "days": "3 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },
    "UKMO": {
        "provider": "UK Met Office",
        "keyword": "ukmo_seamless",
        "country": "United Kingdom",
        "resolution": "2 - 10 km",
        "days": "7 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "KNMI": {
        "provider": "KNMI",
        "keyword": "knmi_seamless",
        "country": "Netherlands",
        "resolution": "2 km",
        "days": "2.5 days",
        "frequency": "Every hour",
        "type": "forecast"
    },
    "DMI": {
        "provider": "DMI",
        "keyword": "dmi_seamless",
        "country": "Denmark",
        "resolution": "2 km",
        "days": "2.5 days",
        "frequency": "Every 3 hours",
        "type": "forecast"
    },

    # ── Archive models ────────────────────────────────────────────────
    "ECMWF IFS": {
        "provider": "ECMWF",
        "keyword": "ecmwf_ifs",
        "country": "European Union",
        "resolution": "9 km",
        "days": "2017 to present",
        "frequency": "Daily with 2 days delay",
        "type": "archive"
    },
    "ERA5": {
        "provider": "ECMWF",
        "keyword": "era5",
        "country": "European Union",
        "resolution": "11 km",
        "days": "1950 to present",
        "frequency": "Daily with 5 days delay",
        "type": "archive"
    },
}
