# 📈 Meteogram generator using Open-Meteo API

This repository contains tools and scripts to generate **meteograms** — graphical weather forecasts — based on meteorological data retrieved from the [Open-Meteo Weather API](https://open-meteo.com/). The project is designed for users who need fast, customizable visualizations of forecast data for specific locations, particularly for fieldwork planning, environmental analysis, scientific communication or educational purposes.

## 🚀 Features

- Query and visualize forecast data from the free Open-Meteo API.
- Multi-source weather comparison: compare current forecasts (ECMWF, GFS, AROME...) with historical conditions from ERA5 for the same dates in previous years.
- Supports hourly and daily forecast variables (temperature, wind, precipitation, etc.).
- Custom meteogram plots using `seaborn` and `pandas`.
- Modular Python code for easy reuse or integration in other projects.
- Script-based generation of meteograms for any location (lat/lon).
- Open-source and well-documented.

## 🛠️ What's Next

Planned features and improvements for future versions:

- Support for ensemble forecast models to visualize prediction spread and uncertainty
- Weather station comparison: overlay observational data from nearby stations to validate forecasts.
- Sounding plots: generate Skew-T diagrams and vertical profiles from forecast or reanalysis data.
- Flask-based API: develop a lightweight and user-friendly API for automated meteogram generation.
- QGIS plugin: integrate meteogram generation into QGIS for spatially-aware forecasting and analysis.

## 🔗 Reference

Zippenfenig, P. *Open-Meteo.com Weather API* [Computer software]. https://doi.org/10.5281/zenodo.7970649
