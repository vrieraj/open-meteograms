#!/usr/bin/env python3
"""
WOW Met Éireann Scraper
========================
Descarga observaciones de una estación WOW Met Éireann usando el endpoint:
  GET https://api.wow.met.ie/table/{siteId}?date=YYYY-MM-DD&startTime=00:00:00&endTime=23:59:59

Características:
- Reanudable: no re-descarga días ya guardados
- Una petición por día
- Salida: CSV con una fila por observación (~5 min de intervalo)
- Rate limiting configurable

Uso:
  python wow_metie_scraper.py
  python wow_metie_scraper.py --start 2025-01-01 --end 2025-12-31
  python wow_metie_scraper.py --site 685c0f31-4740-ee11-805b-201642ba4e29
"""

import requests
import csv
import json
import time
import argparse
import os
from datetime import date, timedelta
from pathlib import Path

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

SITE_ID    = "685c0f31-4740-ee11-805b-201642ba4e29"
SITE_NAME  = "McGovern_NI"          # para el nombre del fichero CSV
START_DATE = date(2024, 5, 14)      # fecha de inicio de la estación
END_DATE   = date.today()           # hasta hoy

OUTPUT_DIR = Path("wow_metie_data")
DELAY_SEC  = 1.0                    # pausa entre peticiones (segundos)

BASE_URL = "https://api.wow.met.ie/table/{site_id}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://wow.met.ie",
    "Referer": "https://wow.met.ie/",
}

# Columnas del CSV de salida
CSV_COLUMNS = [
    "datetime_utc",
    "temp_c",
    "dewpoint_c",
    "humidity_pct",
    "wind_dir_deg",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_gust_dir_deg",
    "rainfall_mm",        # acumulado desde reset (valor del sensor)
    "rainfall_rate_mmh",  # tasa instantánea
    "pressure_hpa",
    "obs_id",
    "external_id",
]

# ─── FUNCIONES ───────────────────────────────────────────────────────────────

def fetch_day(site_id: str, day: date, session: requests.Session) -> list[dict]:
    """Descarga las observaciones de un día. Devuelve lista de dicts."""
    url = BASE_URL.format(site_id=site_id)
    params = {
        "date":      day.strftime("%Y-%m-%d"),
        "startTime": "00:00:00",
        "endTime":   "23:59:59",
    }
    resp = session.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("observations", [])


def parse_observation(obs: dict) -> dict:
    """Extrae los campos relevantes de una observación."""
    return {
        "datetime_utc":       obs.get("reportEndDateTime", ""),
        "temp_c":             obs.get("dryBulbTemperature_Celsius", ""),
        "dewpoint_c":         obs.get("dewPointTemperature_Celsius", ""),
        "humidity_pct":       obs.get("relativeHumidity", ""),
        "wind_dir_deg":       obs.get("windDirection", ""),
        "wind_speed_ms":      obs.get("windSpeed_MetrePerSecond", ""),
        "wind_gust_ms":       obs.get("windGust_MetrePerSecond", ""),
        "wind_gust_dir_deg":  obs.get("windGustDirection", ""),
        "rainfall_mm":        obs.get("rainfallAmount_Millimetre", ""),
        "rainfall_rate_mmh":  obs.get("rainfallRate_MillimetrePerHour", ""),
        "pressure_hpa":       obs.get("meanSeaLevelPressure_Hectopascal", ""),
        "obs_id":             obs.get("id", ""),
        "external_id":        obs.get("externalId", ""),
    }


def days_already_done(csv_path: Path) -> set[str]:
    """Lee el CSV existente y devuelve el conjunto de fechas (YYYY-MM-DD) ya descargadas."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row.get("datetime_utc", "")
            if dt:
                done.add(dt[:10])   # los primeros 10 chars son YYYY-MM-DD
    return done


def daterange(start: date, end: date):
    """Generador de fechas [start, end] inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def run(site_id: str, site_name: str, start: date, end: date, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{site_name}_{site_id[:8]}_observations.csv"

    # Leer progreso anterior
    done = days_already_done(csv_path)
    total_days = (end - start).days + 1
    pending = [d for d in daterange(start, end) if d.strftime("%Y-%m-%d") not in done]

    print(f"Estación : {site_name} ({site_id})")
    print(f"Rango    : {start} → {end}  ({total_days} días)")
    print(f"Pendiente: {len(pending)} días  |  Ya descargados: {len(done)} días")
    print(f"Salida   : {csv_path}")
    print()

    if not pending:
        print("✓ Todo descargado, nada que hacer.")
        return

    # Abrir CSV en modo append
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    session = requests.Session()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, day in enumerate(pending, 1):
            day_str = day.strftime("%Y-%m-%d")
            try:
                observations = fetch_day(site_id, day, session)
                rows_written = 0
                for obs in observations:
                    writer.writerow(parse_observation(obs))
                    rows_written += 1
                f.flush()
                print(f"[{i:4d}/{len(pending)}] {day_str}  →  {rows_written:3d} obs")
            except requests.HTTPError as e:
                print(f"[{i:4d}/{len(pending)}] {day_str}  →  HTTP {e.response.status_code} — saltando")
            except Exception as e:
                print(f"[{i:4d}/{len(pending)}] {day_str}  →  ERROR: {e} — saltando")

            time.sleep(DELAY_SEC)

    print(f"\n✓ Descarga completada → {csv_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    global DELAY_SEC

    parser = argparse.ArgumentParser(description="WOW Met Éireann scraper")
    parser.add_argument("--site",  default=SITE_ID,   help="UUID de la estación")
    parser.add_argument("--name",  default=SITE_NAME, help="Nombre corto para el CSV")
    parser.add_argument("--start", default=str(START_DATE), help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   default=str(END_DATE),   help="Fecha fin YYYY-MM-DD")
    parser.add_argument("--out",   default=str(OUTPUT_DIR), help="Directorio de salida")
    parser.add_argument("--delay", default=DELAY_SEC, type=float, help="Pausa entre requests (s)")
    args = parser.parse_args()

    DELAY_SEC = args.delay

    run(
        site_id    = args.site,
        site_name  = args.name,
        start      = date.fromisoformat(args.start),
        end        = date.fromisoformat(args.end),
        output_dir = Path(args.out),
    )


if __name__ == "__main__":
    main()
