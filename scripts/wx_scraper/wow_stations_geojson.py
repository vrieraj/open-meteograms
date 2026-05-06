#!/usr/bin/env python3
"""
wow_stations_geojson.py
========================
Descarga las estaciones activas de WOW Met Éireann y genera un GeoJSON
con nombre, indicativo (siteId), coordenadas y observación actual.

Hace 2 peticiones (hora actual y hace 7 días) para maximizar
el número de estaciones capturadas.

Uso:
    python wow_stations_geojson.py
    python wow_stations_geojson.py --country IE       # solo Irlanda
    python wow_stations_geojson.py --country GB       # solo UK/NI
    python wow_stations_geojson.py --out my_file.geojson
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json, text/plain, */*",
    "Origin":     "https://wow.met.ie",
    "Referer":    "https://wow.met.ie/",
}
ENDPOINT = "https://api.wow.met.ie/getObservations/dry_bulb"


def probe_slots():
    """Hora actual redondeada a 00/06/12/18, esa -12h, y hace 7 días."""
    now  = datetime.now(timezone.utc).replace(tzinfo=None)
    slot = (now.hour // 6) * 6
    base = now.replace(hour=slot, minute=0, second=0, microsecond=0)
    return [
        (base.strftime("%d/%m/%Y"),            base.hour),
        ((base - timedelta(days=7)).strftime("%d/%m/%Y"),   (base - timedelta(days=7)).hour),
    ]


def fetch_slot(date_str, hour):
    try:
        r = requests.get(
            ENDPOINT,
            params={
                "date":               date_str,
                "time":               hour,
                "timePoint":          -1,
                "mapFilterTags":      "",
                "showWowData":        "true",
                "showOfficialData":   "true",
                "showRegisteredSites":"false",
            },
            headers=HEADERS,
            timeout=20,
        )
        if r.status_code == 200:
            return r.json().get("features", [])
        print(f"  ⚠ {date_str} {hour:02d}h → HTTP {r.status_code}")
    except Exception as e:
        print(f"  ⚠ {date_str} {hour:02d}h → {e}")
    return []


def f2c(f):
    return round((f - 32) * 5 / 9, 1) if f is not None else None


def parse_feature(feat):
    coords = feat.get("geometry", {}).get("coordinates", [None, None])
    lon, lat = coords[0], coords[1]
    p   = feat.get("properties", {})
    pri = p.get("primary", {})

    # Temperatura: la API devuelve °C en dt directamente (verificado)
    temp = round(pri["dt"], 1) if "dt" in pri else None
    hum  = pri.get("dh")
    pres = round(pri["dap"], 1) if "dap" in pri else (
           round(pri["dm"],  1) if "dm"  in pri else None)
    wind_ms = pri.get("dws")
    wind_kmh = round(wind_ms * 3.6, 1) if wind_ms is not None else None

    return {
        "siteId":       str(p.get("siteId") or p.get("msi") or ""),
        "name":         p.get("siteName", ""),
        "lat":          lat,
        "lon":          lon,
        "country":      p.get("country", ""),
        "softwareType": p.get("softwareType", ""),
        "lastReport":   p.get("reportEndDateTime", ""),
        "temp_c":       temp,
        "humidity_pct": hum,
        "pressure_hpa": pres,
        "wind_kmh":     wind_kmh,
    }


def main():
    parser = argparse.ArgumentParser(description="WOW Met Éireann — GeoJSON de estaciones")
    parser.add_argument("--country", "-c", default=None,
                        help="Filtrar por país (ej: IE, GB). Sin filtro = todas.")
    parser.add_argument("--out", "-o", default="wow_stations.geojson",
                        help="Fichero de salida (default: wow_stations.geojson)")
    args = parser.parse_args()

    slots = probe_slots()
    found = {}   # siteId -> station dict

    print(f"Consultando {len(slots)} slots temporales...")
    for date_str, hour in slots:
        print(f"  → {date_str}  {hour:02d}:00 UTC", end=" ", flush=True)
        feats = fetch_slot(date_str, hour)
        new = 0
        for feat in feats:
            s = parse_feature(feat)
            if not s["siteId"] or s["siteId"] in found:
                continue
            if args.country and s["country"].upper() != args.country.upper():
                continue
            found[s["siteId"]] = s
            new += 1
        print(f"(+{new} nuevas, total {len(found)})")
        time.sleep(0.3)

    if not found:
        print("No se encontraron estaciones.")
        return

    # Construir GeoJSON
    features = []
    for s in sorted(found.values(), key=lambda x: x["name"]):
        lat, lon = s["lat"], s["lon"]
        geom = {"type": "Point", "coordinates": [lon, lat]} if lat and lon else None
        props = {k: v for k, v in s.items() if k not in ("lat", "lon")}
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    fc = {"type": "FeatureCollection", "features": features}
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(features)} estaciones → {out}")

    # Preview tabla
    print(f"\n{'siteId':<40} {'Nombre':<35} {'País':<4} {'T°C':>5}  Último reporte")
    print("─" * 100)
    for s in sorted(found.values(), key=lambda x: x["name"]):
        t = f"{s['temp_c']}°C" if s["temp_c"] is not None else "—"
        print(f"{s['siteId']:<40} {s['name'][:34]:<35} {s['country']:<4} {t:>6}  {s['lastReport'][:19]}")


if __name__ == "__main__":
    main()
