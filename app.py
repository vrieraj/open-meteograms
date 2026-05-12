"""Streamlit viewer — Open Meteograms."""

import calendar
import io
import json
import os
from collections import Counter
from dataclasses import dataclass, field
import requests
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox
from datetime import date, timedelta

from scripts.place import Place
from scripts.meteo_sfc import MeteoSfc
from scripts.meteo_vrt import MeteoVrt
from scripts.weather_models import WEATHER_MODELS
from datasources.wx_stations import fetch_wu_stations_near, fetch_wu_hourly

try:
    from astral import LocationInfo
    from astral.sun import sun as astral_sun
    _ASTRAL_OK = True
except ImportError:
    _ASTRAL_OK = False


# ── CONFIG ────────────────────────────────────────────────────
st.set_page_config(page_title="Open Meteograms", layout="wide")


# ── SESSION STATE ─────────────────────────────────────────────
_DEFAULTS = {
    "place": None,
    "map_center": [48, 10],
    "map_zoom": 4,
    "gen_fechas": None,
    "gen_models": None,
    "search_results": {},
    "event": None,
    "open_dialog": False,
    "last_click": None,      # coords (lat, lon) del último click procesado
    "last_selected": None,   # label del último resultado de búsqueda procesado
    # weather stations
    "wx_stations": [],       # list[dict] from fetch_wu_stations_near
    "wx_selected": set(),    # set of stationId strings
    "wu_api_key": os.environ.get("WU_API_KEY", ""),
    "wx_radius_km": 50,
    "wx_searched": False,      # True after a search completes (for empty-result warning)
    # excel download
    "excel_bytes": None,
    "excel_key": None,
    # vector layers
    "geojson_layers": [],  # list[dict]: {name, data, color, visible, n_feat}
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── HELPERS ───────────────────────────────────────────────────
def _fmt_coord(v: float, is_lat: bool = True) -> str:
    hemi = ("N" if v >= 0 else "S") if is_lat else ("E" if v >= 0 else "W")
    return f"{abs(v):.4f}° {hemi}"


def _safe(d: dict, k: str):
    return d.get(k) if isinstance(d, dict) else None


def _sunrise_sunset(place, d: date):
    """Return (sunrise_str, sunset_str) in local time, or (None, None) on error."""
    if not _ASTRAL_OK:
        return None, None
    try:
        loc = LocationInfo(latitude=place.lat, longitude=place.lon,
                           timezone=str(place.tzinfo))
        s = astral_sun(loc.observer, date=d, tzinfo=place.tzinfo)
        return s['sunrise'].strftime('%H:%M'), s['sunset'].strftime('%H:%M')
    except Exception:
        return None, None


def _sidebar_row(label: str, value):
    """Render una fila label / valor en el sidebar."""
    if value in (None, "", "None"):
        return
    c1, c2 = st.columns([1, 1])
    c1.markdown(f"**{label}**")
    c2.markdown(
        f"<div style='text-align:right'>{value}</div>",
        unsafe_allow_html=True,
    )


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _clamp_date(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


_SAT_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_LBL_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)


def _geojson_name_field(data: dict) -> str | None:
    """Return the first property key that looks like a feature name, or None."""
    for feat in (data.get("features") or [])[:10]:
        props = feat.get("properties") or {}
        for candidate in ("name", "Name", "NAME", "label", "nombre", "titulo"):
            if candidate in props:
                return candidate
    return None


def _build_map(center, zoom, basemap, place=None,
               wx_stations=None, wx_selected=None, geojson_layers=None):
    if basemap == "🗺️ Map":
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    elif basemap == "🛰️ Satellite":
        m = folium.Map(location=center, zoom_start=zoom, tiles=None)
        folium.TileLayer(tiles=_SAT_TILES, attr="Esri World Imagery").add_to(m)
    else:  # Hybrid
        m = folium.Map(location=center, zoom_start=zoom, tiles=None)
        folium.TileLayer(tiles=_SAT_TILES, attr="Esri World Imagery").add_to(m)
        folium.TileLayer(
            tiles=_LBL_TILES, attr="Esri Reference",
            overlay=True, control=False,
        ).add_to(m)
    if place:
        folium.Marker([place.lat, place.lon], tooltip=place.name).add_to(m)
    # ── Weather stations ──────────────────────────────────────
    selected = wx_selected or set()
    for s in (wx_stations or []):
        if s.get("lat") is None or s.get("lon") is None:
            continue
        sid = s["stationId"]
        is_sel = sid in selected
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=8 if is_sel else 6,
            color="#ffffff" if is_sel else "#f8a100",
            weight=2 if is_sel else 1,
            fill=True,
            fill_color="#f8a100",
            fill_opacity=1.0 if is_sel else 0.75,
            tooltip=sid,
            popup=folium.Popup(_station_popup(s), max_width=240),
        ).add_to(m)
    # ── GeoJSON vector layers ─────────────────────────────────
    has_layers = False
    for layer in (geojson_layers or []):
        if not layer.get("visible"):
            continue
        color = layer["color"]
        name_field = _geojson_name_field(layer["data"])
        gj = folium.GeoJson(
            layer["data"],
            name=layer["name"],
            style_function=lambda f, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.25,
                "opacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[name_field], labels=False
            ) if name_field else None,
        )
        gj.add_to(m)
        has_layers = True
    if has_layers:
        folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── STATION HELPERS ───────────────────────────────────────────
def _station_labels(stations: list[dict]) -> dict[str, str]:
    """stationId → display label (name or ID, with 01/02 index for duplicates)."""
    raw = {
        s["stationId"]: (((s.get("name") or "").split(",") or [s["stationId"]])[0]).strip() or s["stationId"]
        for s in stations
    }
    counts = Counter(raw.values())
    seen: dict[str, int] = {}
    result: dict[str, str] = {}
    for sid, label in raw.items():
        if counts[label] > 1:
            seen[label] = seen.get(label, 0) + 1
            result[sid] = f"{label} {seen[label]:02d}"
        else:
            result[sid] = label
    return result


@dataclass
class _PlaceLike:
    """Minimal Place substitute for functions that need place attributes."""
    name: str
    lat: float
    lon: float
    elev: float
    tzinfo: str
    properties: dict = field(default_factory=dict)


@st.cache_data(ttl=300, show_spinner=False)
def _build_excel_bytes(name, lat, lon, elev, tzinfo_str,
                        fechas, models_t, wx_sel_t, wu_key, labels_t):
    """Fetch all selected data and return an Excel workbook as bytes."""
    import pandas as pd
    place = _PlaceLike(name=name, lat=lat, lon=lon, elev=elev, tzinfo=tzinfo_str)
    sfc = MeteoSfc(place, list(fechas))
    if models_t:
        sfc.get_data("openmeteo", models=list(models_t))
    labels = dict(labels_t)
    for sid in wx_sel_t:
        df = fetch_wu_hourly(sid, fechas[0], fechas[1], wu_key)
        if df is not None and not df.empty:
            sfc.get_data_station(df, label=labels.get(sid, sid))

    buf = io.BytesIO()
    col_order = ["time", "temperature_2m", "dew_point_2m", "relative_humidity_2m",
                 "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
                 "vapour_pressure_deficit", "fuel_moisture", "fuel_moisture_vpd",
                 "prob_ignition"]
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for src in sfc.datos["source"].unique():
            label = sfc.station_names.get(src, src)
            sheet = label[:31]  # Excel sheet name limit
            df_src = sfc.datos[sfc.datos["source"] == src].copy()
            cols = [c for c in col_order if c in df_src.columns]
            df_src[cols].to_excel(writer, sheet_name=sheet, index=False)
    return buf.getvalue()


# ── WEATHER STATIONS ──────────────────────────────────────────
@st.cache_data(ttl=3_600)
def _cached_wu_stations(lat: float, lon: float, radius_km: float,
                         api_key: str) -> list:
    return fetch_wu_stations_near(lat, lon, radius_km, api_key)


def _station_popup(s: dict) -> str:
    chips = []
    if s.get("temp_c")        is not None: chips.append(f"🌡 {s['temp_c']}°C")
    if s.get("humidity_pct")  is not None: chips.append(f"💧 {s['humidity_pct']}%")
    if s.get("windspeed_kmh") is not None: chips.append(f"💨 {s['windspeed_kmh']} km/h")
    if s.get("pressure_hpa")  is not None: chips.append(f"📊 {s['pressure_hpa']} hPa")
    name = s.get("name") or s.get("stationId", "")
    lines = [f"<b style='font-size:13px'>{name}</b>",
             f"<code style='font-size:10px;color:#888'>{s['stationId']}</code>"]
    if s.get("adm1"):    lines.append(f"<small>📍 {s['adm1']}</small>")
    if s.get("elev_m") is not None: lines.append(f"<small>⛰ {s['elev_m']} m</small>")
    if chips: lines.append("<small>" + "  " + "  ".join(chips) + "</small>")
    wu_url = f"https://www.wunderground.com/dashboard/pws/{s['stationId']}"
    lines.append(f"<small><a href='{wu_url}' target='_blank'>View on Wunderground →</a></small>")
    return "<br>".join(lines)


# ── NOMINATIM ─────────────────────────────────────────────────
@st.cache_data(ttl=86_400)
def nominatim_search(q: str) -> tuple[dict, list]:
    if not q or len(q) < 3:
        return {}, []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"format": "geocodejson", "q": q, "limit": 5,
                    "addressdetails": 1, "namedetails": 1},
            headers={"User-Agent": "OpenMeteograms/1.0"},
            timeout=5,
        )
        feats = r.json().get("features", [])
        results, labels = {}, []
        for f in feats:
            lbl = f["properties"]["geocoding"]["label"]
            results[lbl] = f
            labels.append(lbl)
        return results, labels
    except Exception:
        return {}, []


@st.cache_data(ttl=86_400)
def nominatim_reverse(lat: float, lon: float) -> list:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "geocodejson", "lat": lat, "lon": lon},
            headers={"User-Agent": "OpenMeteograms/1.0"},
            timeout=5,
        )
        return r.json().get("features", [])
    except Exception:
        return []


def _search_fn(q: str) -> list[str]:
    results, labels = nominatim_search(q)
    # Solo actualizar si hay resultados: st_searchbox puede llamar a esta
    # función con query vacío durante reruns, lo que borraría los resultados
    # del último label seleccionado antes de que podamos recuperar el feature.
    if results:
        st.session_state.search_results = results
    return labels


# ── HEADER ────────────────────────────────────────────────────
col_search, col_spacer, col_map = st.columns([3, 1, 2])

with col_search:
    selected = st_searchbox(
        _search_fn,
        placeholder="Search location...",
        key="searchbox_widget",
    )

with col_map:
    _basemap = st.radio(
        "Basemap",
        ["🗺️ Map", "🛰️ Satellite", "🌍 Hybrid"],
        horizontal=True,
        key="basemap",
        label_visibility="collapsed",
    )
with st.sidebar:
    st.markdown("""
    <div style="
        font-size:20px;
        font-weight:700;
        margin-bottom:8px;
    ">
    🌤️ Open Meteograms
    </div>
    """, unsafe_allow_html=True)

# st_searchbox persiste el valor entre reruns igual que st_folium con
# last_clicked. Sin deduplicación: rerun detecta selected → encola evento
# → rerun → loop infinito. Comparar contra last_selected lo evita.
if isinstance(selected, str) and selected != st.session_state.last_selected:
    feat = st.session_state.search_results.get(selected)
    if feat:
        st.session_state.last_selected = selected
        st.session_state.event = {"type": "search", "feature": feat}
        st.rerun()


# ── EVENT PROCESSOR ───────────────────────────────────────────
# Debe ejecutarse ANTES del render del mapa para que map_center y
# place estén actualizados cuando folium construya el mapa.
_ev = st.session_state.event
if _ev:
    st.session_state.event = None   # consumir antes de procesar (evita loops)

    if _ev["type"] == "search":
        try:
            p = Place(_ev["feature"])
            st.session_state.place = p
            st.session_state.map_center = [p.lat, p.lon]
            st.session_state.map_zoom = 12
            # No resetear last_click aquí: si lo ponemos a None, el click
            # anterior del mapa vuelve a dispararse en el siguiente rerun
            # (st_folium lo persiste), lo que roba el rerun al dialog.
        except Exception:
            st.warning("⚠️ Localización no soportada.")

    elif _ev["type"] == "map":
        feats = nominatim_reverse(*_ev["coords"])
        if feats:
            try:
                st.session_state.place = Place(feats[0])
                st.session_state.map_center = list(_ev["coords"])
                st.session_state.map_zoom = 12
            except Exception:
                st.warning("⚠️ Geocodificación inversa fallida.")
        else:
            st.warning("⚠️ No se encontró ninguna localización en ese punto.")

# ── MAP ───────────────────────────────────────────────────────
_m = _build_map(
    st.session_state.map_center,
    st.session_state.map_zoom,
    _basemap,
    st.session_state.place,
    wx_stations=st.session_state.wx_stations,
    wx_selected=st.session_state.wx_selected,
    geojson_layers=st.session_state.geojson_layers,
)

map_data = st_folium(
    _m,
    use_container_width=True,
    height=650,
    returned_objects=["last_clicked"],
)

# Detectar click y encolar evento solo si las coordenadas son nuevas.
# st_folium persiste last_clicked entre reruns, por lo que sin esta
# deduplicación se generaría un bucle infinito.
_click = (map_data or {}).get("last_clicked")
if _click:
    _coords = (round(_click["lat"], 5), round(_click["lng"], 5))
    if _coords != st.session_state.last_click:
        st.session_state.last_click = _coords
        # Station click: toggle selection instead of geocoding
        _hit = next(
            (s["stationId"] for s in st.session_state.wx_stations
             if abs(s["lat"] - _coords[0]) < 0.001
             and abs(s["lon"] - _coords[1]) < 0.001),
            None,
        )
        if _hit:
            _sel = set(st.session_state.wx_selected)
            _sel.discard(_hit) if _hit in _sel else _sel.add(_hit)
            st.session_state.wx_selected = _sel
        else:
            st.session_state.event = {"type": "map", "coords": _coords}
        st.rerun()


# ── DIALOG ────────────────────────────────────────────────────
# Definido antes del sidebar para poder llamarlo directamente desde el botón.
@st.dialog("Meteogram", width="large")
def _dialog():
    place  = st.session_state.place
    fechas = st.session_state.gen_fechas
    models = st.session_state.gen_models

    if not place or not fechas or not models:
        st.error("Faltan datos de entrada.")
        return

    try:
        with st.spinner("Generando meteograma..."):
            sfc = MeteoSfc(place, fechas)
            sfc.get_data("openmeteo", models=models)
            vrt = None
            if (len(models) == 1
                    and WEATHER_MODELS.get(models[0], {}).get('type') == 'forecast'):
                vrt = MeteoVrt(place, fechas)
                vrt.get_data("openmeteo", model=models[0])

        wx_selected = st.session_state.get("wx_selected", set())
        wu_key      = st.session_state.get("wu_api_key", "").strip()
        if wx_selected and wu_key:
            _lbl_map = _station_labels(st.session_state.get("wx_stations", []))
            for sid in wx_selected:
                with st.spinner(f"Downloading station {_lbl_map.get(sid, sid)}…"):
                    df = fetch_wu_hourly(sid, fechas[0], fechas[1], wu_key)
                if df is not None and not df.empty:
                    sfc.get_data_station(df, label=_lbl_map.get(sid, sid))
                else:
                    st.warning(f"⚠ No data for {_lbl_map.get(sid, sid)} in this date range.")

        tab_meteo, tab_skewt = st.tabs(["🌤 Meteogram", "📡 Skew-T"])

        with tab_meteo:
            with st.spinner("Rendering…"):
                fig = sfc.meteoplot(vrt=vrt)
                st.pyplot(fig)

        with tab_skewt:
            import matplotlib.pyplot as _plt

            skewt_models = [m for m in models
                            if WEATHER_MODELS.get(m, {}).get('type') in ('forecast', 'archive')]

            if not skewt_models:
                st.info("Skew-T requires at least one forecast or archive model.")
            else:
                if len(skewt_models) > 1:
                    sel_model = st.selectbox("Model", skewt_models, key="skewt_model_sel")
                else:
                    sel_model = skewt_models[0]
                    st.caption(f"Model: **{sel_model}**")

                _cache_key = ("skewt_vrt", sel_model, fechas[0], fechas[1],
                              place.lat, place.lon)

                if _cache_key not in st.session_state:
                    with st.spinner(f"Fetching {sel_model} vertical data…"):
                        vrt_s = MeteoVrt(place, fechas)
                        vrt_s.get_data('openmeteo', model=sel_model)
                        vrt_s._fetch_skewt_data()
                    st.session_state[_cache_key] = vrt_s

                vrt_s = st.session_state[_cache_key]

                if vrt_s._datos_skewt is None:
                    st.warning("No vertical data available for this model.")
                else:
                    times = vrt_s._datos_skewt['time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
                    sel_time = st.select_slider("🕐 Hour", options=times, key="skewt_time_sel")

                    col_skewt, col_idx = st.columns([3, 1])

                    with col_idx:
                        st.markdown("**Índices**")
                        indices = vrt_s.compute_skewt_indices(sel_time)
                        if indices:
                            st.metric("CAPE", f"{indices['cape']:.0f} J/kg")
                            st.metric("CIN", f"{indices['cin']:.0f} J/kg")
                            st.metric("LCL", f"{indices['lcl_hpa']:.0f} hPa")
                            st.metric("LCL T", f"{indices['lcl_temp']:.1f} °C")
                            if indices.get('trigger_temp') is not None:
                                st.metric("T disparo", f"{indices['trigger_temp']:.1f} °C")
                        else:
                            st.caption("—")

                    with col_skewt:
                        with st.spinner("Rendering Skew-T…"):
                            try:
                                fig_st = vrt_s.skewt(sel_time)
                                st.pyplot(fig_st)
                                _plt.close(fig_st)
                            except Exception as e_st:
                                st.error(f"Skew-T error: {e_st}")
    except Exception as e:
        st.error(str(e))


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    place = st.session_state.place

    # ── LOCATION INFO ─────────────────────────────────────────
    with st.expander("📍 Location info", expanded=True):
        if not place:
            st.info("🗺️ Search a location or click the map.")
        else:
            props = getattr(place, "properties", {}) or {}
            _sr, _ss = _sunrise_sunset(place, date.today())
            _sun_row = (
                f'<div style="display:flex; justify-content:space-between;">'
                f'<b>Sunrise / Sunset</b>'
                f'<span>{_sr} / {_ss}</span></div>'
                if _sr else ''
            )
            st.markdown(f"""
            <div style="line-height:1.25; font-size:15px">

            <div style="display:flex; justify-content:space-between;">
              <b>Name</b><span>{place.name}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>State</b><span>{_safe(props, "state")}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>County</b><span>{_safe(props, "county")}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>Country</b><span>{_safe(props, "country")}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>Timezone</b><span>{place.tzinfo} (UTC{place.delta_time:+d})</span>
            </div>

            {_sun_row}

            <hr style="margin:6px 0; opacity:0.3">

            <div style="display:flex; justify-content:space-between;">
              <b>Latitude</b><span>{place.lat:.4f}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>Longitude</b><span>{place.lon:.4f}</span>
            </div>

            <div style="display:flex; justify-content:space-between;">
              <b>Elevation</b><span>{place.elev if place.elev else "N/A"} m</span>
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style='text-align:center;margin-top:8px'>
                    <a href="{place.map}" target="_blank">Maps</a> ·
                    <a href="{place.meteo['windy']}" target="_blank">Windy</a> ·
                    <a href="{place.meteo['meteoblue']}" target="_blank">Meteoblue</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── VECTOR LAYERS ─────────────────────────────────────────
    with st.expander("📂 Vector layers", expanded=False):
        uploaded = st.file_uploader(
            "Upload GeoJSON",
            type=["geojson", "json"],
            key="geojson_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                raw = json.loads(uploaded.read())
                if raw.get("type") == "Feature":
                    raw = {"type": "FeatureCollection", "features": [raw]}
                n_feat = len(raw.get("features") or [])
                fname = uploaded.name.rsplit(".", 1)[0]
                existing = [l["name"] for l in st.session_state.geojson_layers]
                if fname not in existing:
                    st.session_state.geojson_layers.append({
                        "name":    fname,
                        "data":    raw,
                        "color":   "#e05c00",
                        "visible": True,
                        "n_feat":  n_feat,
                    })
                    st.success(f"Loaded **{fname}** — {n_feat} features")
            except Exception as exc:
                st.error(f"Invalid GeoJSON: {exc}")

        layers = st.session_state.geojson_layers
        to_remove = None
        for i, layer in enumerate(layers):
            col_info, col_col, col_del = st.columns([4, 1, 0.5])
            col_info.markdown(f"**{layer['name']}**")
            col_info.caption(f"{layer['n_feat']} features")
            layer["color"] = col_col.color_picker(
                "", value=layer["color"],
                key=f"vec_col_{i}", label_visibility="collapsed"
            )
            if col_del.button("✕", key=f"vec_del_{i}"):
                to_remove = i
        if to_remove is not None:
            st.session_state.geojson_layers.pop(to_remove)
            st.rerun()

    st.divider()

    if place:
        today = date.today()

        # ── DATE RANGE ────────────────────────────────────────
        with st.expander("📅 Date range", expanded=True):
            default_end = today + timedelta(days=7)

            st.caption("**Start**")
            c1, c2 = st.columns([1, 2])
            s_day   = c1.number_input("Day",   1, 31, value=today.day,           key="s_day")
            s_month = c2.selectbox("Month", MONTHS, index=today.month - 1,       key="s_month")

            st.caption("**End**")
            c3, c4 = st.columns([1, 2])
            e_day   = c3.number_input("Day",   1, 31, value=default_end.day,     key="e_day")
            e_month = c4.selectbox("Month", MONTHS, index=default_end.month - 1, key="e_month")

            current_year = today.year
            years = list(range(current_year + 1, 1979, -1))
            year  = st.selectbox("Year", years, index=years.index(current_year), key="year_sel")

            s_m = MONTHS.index(s_month) + 1
            e_m = MONTHS.index(e_month) + 1
            # Solo cruce de año permitido: diciembre → enero
            e_y = year + 1 if (s_m == 12 and e_m == 1) else year

            d_start = _clamp_date(year, s_m, int(s_day))
            d_end   = _clamp_date(e_y,  e_m, int(e_day))

            delta_days = (d_end - d_start).days
            valid_dates = True
            if delta_days < 0:
                st.error("❌ End date must be after start date.")
                valid_dates = False
            elif delta_days > 10:
                d_end = d_start + timedelta(days=10)
                st.warning(f"⚠️ Range clamped to 10 days → {d_end}")
            else:
                st.caption(f"📆 {d_start} → {d_end}")

        # ── WEATHER MODELS ────────────────────────────────────
        with st.expander("🌐 Weather models", expanded=True):
            raw_models = []
            for i in range(4):
                model = st.selectbox(
                    f"Model {i + 1}",
                    ["—"] + list(WEATHER_MODELS.keys()),
                    key=f"model_{i}",
                )
                raw_models.append(model)
                if model != "—":
                    info = WEATHER_MODELS[model]
                    st.caption(f"🏢 **{info['provider']}** · {info['country']}")
                    st.caption(
                        f"📐 {info['resolution']} · "
                        f"⏱ {info['frequency']} · "
                        f"📅 {info['days']}"
                    )
            selected_models = list(dict.fromkeys(m for m in raw_models if m != "—"))

        # ── WEATHER STATIONS ──────────────────────────────────
        with st.expander("📡 Weather stations", expanded=False):
            wu_key = st.text_input(
                "WU API Key",
                value=st.session_state.wu_api_key,
                type="password",
                key="wu_api_key_input",
                placeholder="e1f10a1e78da46f5…",
                help="Set WU_API_KEY in your environment to pre-fill on startup.",
            )
            st.session_state.wu_api_key = wu_key

            radius_km = st.slider(
                "Search radius (km)", 10, 200,
                value=st.session_state.wx_radius_km,
                step=10, key="wx_radius_slider",
            )
            st.session_state.wx_radius_km = radius_km

            search_disabled = not wu_key.strip()
            if st.button("🔍 Search nearby stations",
                         disabled=search_disabled,
                         use_container_width=True):
                with st.spinner("Searching stations…"):
                    found = _cached_wu_stations(
                        place.lat, place.lon, radius_km, wu_key.strip()
                    )
                st.session_state.wx_stations = found
                st.session_state.wx_selected = set()
                st.session_state.wx_searched = True
                st.rerun()

            stations = st.session_state.wx_stations
            if st.session_state.wx_searched and not stations:
                st.warning("No stations found. Try increasing the radius.")
            if stations:
                labels = _station_labels(stations)
                st.caption(f"{len(stations)} stations found")
                st.divider()
                new_selected = set(st.session_state.wx_selected)
                for s in stations:
                    sid   = s["stationId"]
                    label = labels[sid]
                    loc   = " · ".join(filter(None, [s.get("adm1"), s.get("country")]))
                    elev  = f"  ⛰ {s['elev_m']} m" if s.get("elev_m") is not None else ""
                    col_cb, col_info = st.columns([1, 5])
                    checked = col_cb.checkbox("", value=sid in new_selected,
                                              key=f"wx_cb_{sid}")
                    col_info.markdown(f"**{label}**")
                    col_info.caption(f"`{sid}`  {loc}{elev}")
                    if checked:
                        new_selected.add(sid)
                    else:
                        new_selected.discard(sid)
                st.session_state.wx_selected = new_selected
                if new_selected:
                    st.caption(f"✓ {len(new_selected)} station(s) selected for meteogram")

        # ── GENERATE ──────────────────────────────────────────
        st.divider()
        _can_run = bool(selected_models) and valid_dates
        if st.button(
            "⚡ Generate Meteogram",
            type="primary",
            use_container_width=True,
            disabled=not _can_run,
        ):
            st.session_state.gen_fechas = [str(d_start), str(d_end)]
            st.session_state.gen_models = selected_models
            st.session_state.open_dialog = True

        # ── DOWNLOAD EXCEL ────────────────────────────────────
        _wx_stations = st.session_state.wx_stations
        _wx_selected = st.session_state.wx_selected
        _wu_key      = st.session_state.wu_api_key.strip()
        _labels      = _station_labels(_wx_stations) if _wx_stations else {}
        _excel_key   = (
            tuple(selected_models),
            tuple(sorted(_wx_selected)),
            str(d_start), str(d_end),
            _wu_key,
        )
        if _excel_key != st.session_state.excel_key:
            st.session_state.excel_bytes = None
            st.session_state.excel_key   = _excel_key

        _dl_disabled = not _can_run or (not selected_models and not _wx_selected)
        if st.button("📥 Prepare data download",
                     use_container_width=True,
                     disabled=_dl_disabled):
            with st.spinner("Fetching data…"):
                st.session_state.excel_bytes = _build_excel_bytes(
                    place.name, place.lat, place.lon,
                    place.elev, str(place.tzinfo),
                    (str(d_start), str(d_end)),
                    tuple(selected_models),
                    tuple(sorted(_wx_selected)),
                    _wu_key,
                    tuple(_labels.items()),
                )
        if st.session_state.excel_bytes:
            st.download_button(
                "⬇ Download Excel",
                data=st.session_state.excel_bytes,
                file_name=f"meteogram_{place.name}_{d_start}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:12px'>"
        "<a href='https://github.com/vrieraj/open-meteograms' target='_blank'>"
        "About · GitHub</a></div>",
        unsafe_allow_html=True,
    )

# ── DIALOG (fuera del sidebar) ────────────────────────────────
if st.session_state.open_dialog:
    st.session_state.open_dialog = False
    _dialog()