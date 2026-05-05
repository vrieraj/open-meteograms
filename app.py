"""Streamlit viewer — Open Meteograms."""

import calendar
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox
from datetime import date, timedelta

from scripts.place import Place
from scripts.meteo_sfc import MeteoSfc
from scripts.meteo_vrt import MeteoVrt
from scripts.weather_models import WEATHER_MODELS


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
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── HELPERS ───────────────────────────────────────────────────
def _fmt_coord(v: float, is_lat: bool = True) -> str:
    hemi = ("N" if v >= 0 else "S") if is_lat else ("E" if v >= 0 else "W")
    return f"{abs(v):.4f}° {hemi}"


def _safe(d: dict, k: str):
    return d.get(k) if isinstance(d, dict) else None


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


def _build_map(center, zoom, basemap, place=None):
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
    return m


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
            if len(models) == 1:
                vrt = MeteoVrt(place, fechas)
                vrt.get_data("openmeteo", model=models[0])
            fig = sfc.meteoplot(vrt=vrt)
            st.pyplot(fig)
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

        # ── GENERATE ──────────────────────────────────────────
        st.divider()
        if st.button(
            "⚡ Generate Meteogram",
            type="primary",
            use_container_width=True,
            disabled=not selected_models or not valid_dates,
        ):
            st.session_state.gen_fechas = [str(d_start), str(d_end)]
            st.session_state.gen_models = selected_models
            st.session_state.open_dialog = True

# ── DIALOG (fuera del sidebar) ────────────────────────────────
if st.session_state.open_dialog:
    st.session_state.open_dialog = False
    _dialog()