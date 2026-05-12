/* ── WEATHER MODELS ───────────────────────────────────────────────────────── */
const WEATHER_MODELS = {
  'ICON':       { provider: 'DWD (Germany)',          type: 'forecast', resolution: '2–11 km' },
  'GFS':        { provider: 'NOAA (USA)',              type: 'forecast', resolution: '3–25 km' },
  'AROME':      { provider: 'MeteoFrance',             type: 'forecast', resolution: '1–25 km' },
  'IFS':        { provider: 'ECMWF',                   type: 'forecast', resolution: '25 km' },
  'GSM JMA':    { provider: 'JMA (Japan)',              type: 'forecast', resolution: '5–55 km' },
  'MET Nordic': { provider: 'MET Norway',              type: 'forecast', resolution: '1 km' },
  'GEM':        { provider: 'Canadian Weather Service',type: 'forecast', resolution: '2.5 km' },
  'GFS GRAPES': { provider: 'CMA (China)',              type: 'forecast', resolution: '15 km' },
  'ACCESS-G':   { provider: 'BOM (Australia)',         type: 'forecast', resolution: '15 km' },
  'COSMO':      { provider: 'ARPAE ARPAP (Italy)',     type: 'forecast', resolution: '2 km' },
  'UKMO':       { provider: 'UK Met Office',           type: 'forecast', resolution: '2–10 km' },
  'KNMI':       { provider: 'KNMI (Netherlands)',      type: 'forecast', resolution: '2 km' },
  'DMI':        { provider: 'DMI (Denmark)',           type: 'forecast', resolution: '2 km' },
  'ECMWF IFS':  { provider: 'ECMWF',                   type: 'archive',  resolution: '9 km' },
  'ERA5':       { provider: 'ECMWF',                   type: 'archive',  resolution: '11 km' },
};

/* ── STATE ──────────────────────────────────────────────────────────────────── */
const state = {
  place: null,
  stations: [],
  selectedStations: new Set(),
  geojsonLayers: [],
};

let _modalPayload = null;
let _modalBaseRange = null;

/* ── MAP ─────────────────────────────────────────────────────────────────────────── */
const TILES = {
  map:        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                { attribution: '© CartoDB', maxZoom: 19 }),
  satellite:  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                { attribution: '© Esri', maxZoom: 19 }),
  hybrid_sat: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                { attribution: '© Esri', maxZoom: 19 }),
  hybrid_lbl: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                { attribution: '© Esri', maxZoom: 19 }),
};

const map = L.map('map').setView([48, 10], 4);
TILES.map.addTo(map);

const locationMarker = L.marker([0, 0]);
const stationLayer   = L.layerGroup().addTo(map);

function switchBasemap(name) {
  Object.values(TILES).forEach(t => { if (map.hasLayer(t)) map.removeLayer(t); });
  if (name === 'map') {
    TILES.map.addTo(map);
  } else if (name === 'satellite') {
    TILES.satellite.addTo(map);
  } else {
    TILES.hybrid_sat.addTo(map);
    TILES.hybrid_lbl.addTo(map);
  }
  document.querySelectorAll('.basemap-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.basemap === name));
}

document.querySelectorAll('.basemap-btn').forEach(btn =>
  btn.addEventListener('click', () => switchBasemap(btn.dataset.basemap)));

/* ── SEARCH ────────────────────────────────────────────────────────────────────────────── */
let searchTimeout = null;
const searchInput    = document.getElementById('search-input');
const searchDropdown = document.getElementById('search-dropdown');

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimeout);
  const q = searchInput.value.trim();
  if (q.length < 3) { hideDropdown(); return; }
  searchTimeout = setTimeout(() => doSearch(q), 300);
});

searchInput.addEventListener('keydown', e => {
  if (e.key === 'Escape') hideDropdown();
});

document.addEventListener('click', e => {
  if (!e.target.closest('#search-container')) hideDropdown();
});

async function doSearch(q) {
  try {
    const res  = await fetch(
      `https://nominatim.openstreetmap.org/search?format=geocodejson&q=${encodeURIComponent(q)}&limit=5&addressdetails=1&namedetails=1&accept-language=en`
    );
    const data = await res.json();
    const results = (data.features || []).map(f => ({
      label: f.properties.geocoding.label,
      lat:   f.geometry.coordinates[1],
      lon:   f.geometry.coordinates[0],
    }));
    renderDropdown(results);
  } catch { hideDropdown(); }
}

function renderDropdown(results) {
  searchDropdown.innerHTML = '';
  if (!results.length) { hideDropdown(); return; }
  results.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r.label;
    li.addEventListener('click', () => selectResult(r));
    searchDropdown.appendChild(li);
  });
  searchDropdown.classList.remove('hidden');
}

function hideDropdown() { searchDropdown.classList.add('hidden'); }

async function selectResult(r) {
  hideDropdown();
  searchInput.value = r.label;
  await loadPlace(r.lat, r.lon, r.label);
  map.setView([r.lat, r.lon], 12);
}

/* ── MAP CLICK ────────────────────────────────────────────────────────────────────────────── */
map.on('click', async (e) => {
  const { lat, lng } = e.latlng;
  try {
    const res  = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=geocodejson&lat=${lat}&lon=${lng}`
    );
    const data = await res.json();
    const feats = data.features || [];
    if (!feats.length) return;
    const label = feats[0].properties.geocoding.label || '';
    await loadPlace(lat, lng, label);
  } catch {}
});

/* ── PLACE ────────────────────────────────────────────────────────────────────────────── */
async function loadPlace(lat, lon, name) {
  try {
    const res   = await fetch(`/api/place?lat=${lat}&lon=${lon}&name=${encodeURIComponent(name)}`);
    const place = await res.json();
    if (place.error) { console.error(place.error); return; }
    state.place = place;
    renderLocationPanel(place);
    if (!map.hasLayer(locationMarker)) locationMarker.addTo(map);
    locationMarker.setLatLng([place.lat, place.lon]).bindTooltip(place.name);
    updateGenerateBtn();
  } catch (e) { console.error(e); }
}

function renderLocationPanel(p) {
  const sunRows = (p.sunrise && p.sunset)
    ? `<tr><td>Sunrise</td><td>${p.sunrise}</td></tr>
       <tr><td>Sunset</td><td>${p.sunset}</td></tr>`
    : '';
  document.getElementById('location-content').innerHTML = `
    <table class="info-table">
      <tr><td>Name</td><td>${escHtml(p.name)}</td></tr>
      <tr><td>Lat</td><td>${p.lat.toFixed(4)}°</td></tr>
      <tr><td>Lon</td><td>${p.lon.toFixed(4)}°</td></tr>
      <tr><td>Elevation</td><td>${p.elev ?? 'N/A'} m</td></tr>
      <tr><td>Timezone</td><td>${escHtml(p.tz)} (UTC${p.delta_time >= 0 ? '+' : ''}${p.delta_time})</td></tr>
      ${sunRows}
    </table>
    <div class="ext-links">
      <a href="${p.map}" target="_blank">Maps</a>
      <a href="${p.urls.windy}" target="_blank">Windy</a>
      <a href="${p.urls.meteoblue}" target="_blank">Meteoblue</a>
    </div>`;
}

/* ── STATIONS ────────────────────────────────────────────────────────────────────────────── */
document.getElementById('radius-slider').addEventListener('input', function () {
  document.getElementById('radius-val').textContent = this.value;
});

document.getElementById('search-stations-btn').addEventListener('click', async () => {
  if (!state.place) return;
  const apiKey = document.getElementById('wu-key').value.trim();
  if (!apiKey) return;
  const radius = document.getElementById('radius-slider').value;
  const btn    = document.getElementById('search-stations-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Searching…';
  try {
    const res      = await fetch(`/api/stations?lat=${state.place.lat}&lon=${state.place.lon}&radius_km=${radius}&api_key=${encodeURIComponent(apiKey)}`);
    const stations = await res.json();
    if (!Array.isArray(stations)) throw new Error(stations.error || 'Unknown error');
    state.stations = stations;
    state.selectedStations.clear();
    renderStationMarkers();
    renderStationsList();
  } catch (e) {
    alert('Station search failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Search stations';
  }
});

function renderStationMarkers() {
  stationLayer.clearLayers();
  state.stations.forEach(s => {
    if (s.lat == null || s.lon == null) return;
    const marker = L.circleMarker([s.lat, s.lon], markerStyle(s.stationId));
    marker.bindPopup(stationPopupHtml(s));
    marker.on('click', e => {
      L.DomEvent.stopPropagation(e);
      toggleStation(s.stationId);
    });
    marker.addTo(stationLayer);
    s._marker = marker;
  });
}

function markerStyle(id) {
  const sel = state.selectedStations.has(id);
  return { radius: sel ? 8 : 6, color: '#fff', weight: sel ? 2 : 1,
           fillColor: '#f8a100', fillOpacity: sel ? 1.0 : 0.75 };
}

function toggleStation(id) {
  state.selectedStations.has(id)
    ? state.selectedStations.delete(id)
    : state.selectedStations.add(id);
  state.stations.forEach(s => s._marker?.setStyle(markerStyle(s.stationId)));
  renderStationsList();
  updateGenerateBtn();
}

function stationPopupHtml(s) {
  const chips = [];
  if (s.temp_c        != null) chips.push(`🌡 ${s.temp_c}°C`);
  if (s.humidity_pct  != null) chips.push(`💧 ${s.humidity_pct}%`);
  if (s.windspeed_kmh != null) chips.push(`💨 ${s.windspeed_kmh} km/h`);
  const sel = state.selectedStations.has(s.stationId);
  return `<b>${escHtml(s.name || s.stationId)}</b><br>
    <code style="font-size:10px;color:#888">${s.stationId}</code><br>
    ${s.adm1 ? `<small>📍 ${escHtml(s.adm1)}</small><br>` : ''}
    ${s.elev_m != null ? `<small>⛰ ${s.elev_m} m</small><br>` : ''}
    ${chips.length ? `<small>${chips.join('  ')}</small><br>` : ''}
    <small><a href="https://www.wunderground.com/dashboard/pws/${s.stationId}" target="_blank">View on Wunderground →</a></small><br>
    <button style="margin-top:5px;font-size:11px;padding:2px 8px;cursor:pointer"
      onclick="toggleStation('${s.stationId}');this.textContent=state.selectedStations.has('${s.stationId}')?'✓ Selected':'+ Select'">
      ${sel ? '✓ Selected' : '+ Select'}
    </button>`;
}

function renderStationsList() {
  const el = document.getElementById('stations-list');
  if (!state.stations.length) { el.innerHTML = ''; return; }
  el.innerHTML = `<div class="station-count">${state.stations.length} stations · ${state.selectedStations.size} selected</div>`;
  state.stations.forEach(s => {
    const sel = state.selectedStations.has(s.stationId);
    const div = document.createElement('div');
    div.className = `station-item${sel ? ' selected' : ''}`;
    div.innerHTML = `<label>
      <input type="checkbox" ${sel ? 'checked' : ''}>
      <span><span class="station-name">${escHtml((s.name || s.stationId).split(',')[0])}</span>
      <span class="station-id">${s.stationId}</span></span>
    </label>`;
    div.querySelector('input').addEventListener('change', () => toggleStation(s.stationId));
    el.appendChild(div);
  });
}

/* ── GEOJSON LAYERS ────────────────────────────────────────────────────────────────────────────── */
document.getElementById('geojson-upload').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      let data = JSON.parse(ev.target.result);
      if (data.type === 'Feature') data = { type: 'FeatureCollection', features: [data] };
      const name = file.name.replace(/\.(geojson|json)$/i, '');
      if (state.geojsonLayers.find(l => l.name === name)) return;
      const color  = '#e05c00';
      const geoLayer = L.geoJSON(data, {
        style: { color, fillColor: color, weight: 2, fillOpacity: 0.25, opacity: 0.85 },
      }).addTo(map);
      state.geojsonLayers.push({ name, data, color, layer: geoLayer });
      map.fitBounds(geoLayer.getBounds(), { padding: [20, 20] });
      renderLayersList();
    } catch (err) { alert('Invalid GeoJSON: ' + err.message); }
  };
  reader.readAsText(file);
  this.value = '';
});

function renderLayersList() {
  const el = document.getElementById('layers-list');
  el.innerHTML = '';
  state.geojsonLayers.forEach((lyr, i) => {
    const div = document.createElement('div');
    div.className = 'layer-item';
    div.innerHTML = `<span title="${escHtml(lyr.name)}">${escHtml(lyr.name)}</span>
      <input type="color" value="${lyr.color}">
      <button title="Remove">✕</button>`;
    div.querySelector('input[type="color"]').addEventListener('input', function () {
      lyr.color = this.value;
      lyr.layer.setStyle({ color: lyr.color, fillColor: lyr.color });
    });
    div.querySelector('button').addEventListener('click', () => {
      map.removeLayer(lyr.layer);
      state.geojsonLayers.splice(i, 1);
      renderLayersList();
    });
    el.appendChild(div);
  });
}

/* ── MODELS ────────────────────────────────────────────────────────────────────────────── */
function initModelSelects() {
  const container = document.getElementById('models-selects');
  Object.keys(WEATHER_MODELS).forEach((_, i) => {
    if (i >= 4) return;
    const wrap = document.createElement('div');
    wrap.className = 'model-row';
    const sel  = document.createElement('select');
    sel.id = `model-${i}`;
    sel.innerHTML = `<option value="">— Model ${i + 1} —</option>` +
      Object.keys(WEATHER_MODELS).map(m => `<option value="${m}">${m}</option>`).join('');
    const info = document.createElement('div');
    info.className = 'model-info';
    info.id = `model-info-${i}`;
    sel.addEventListener('change', () => {
      const m = WEATHER_MODELS[sel.value];
      info.textContent = m ? `${m.provider} · ${m.resolution}` : '';
      updateGenerateBtn();
    });
    wrap.appendChild(sel);
    wrap.appendChild(info);
    container.appendChild(wrap);
  });
}

function getSelectedModels() {
  return [...new Set(
    [0, 1, 2, 3]
      .map(i => document.getElementById(`model-${i}`)?.value)
      .filter(Boolean)
  )];
}

/* ── DATES ────────────────────────────────────────────────────────────────────────────── */
function initDates() {
  const today = new Date();
  const end   = new Date(today);
  end.setDate(end.getDate() + 7);
  document.getElementById('date-start').value = fmtDate(today);
  document.getElementById('date-end').value   = fmtDate(end);
  document.getElementById('date-start').addEventListener('change', validateDates);
  document.getElementById('date-end').addEventListener('change', validateDates);
}

function fmtDate(d) { return d.toISOString().split('T')[0]; }

function validateDates() {
  const start = document.getElementById('date-start').value;
  const end   = document.getElementById('date-end').value;
  const warn  = document.getElementById('date-warning');
  if (!start || !end) return;
  const diff = (new Date(end) - new Date(start)) / 86400000;
  if (diff < 0) {
    warn.textContent = '❌ End must be after start';
    warn.className = 'error';
  } else if (diff > 10) {
    const clamped = new Date(start);
    clamped.setDate(clamped.getDate() + 10);
    document.getElementById('date-end').value = fmtDate(clamped);
    warn.textContent = '⚠️ Range clamped to 10 days';
    warn.className = 'warning';
  } else {
    warn.textContent = '';
    warn.className = '';
  }
}

/* ── COLLAPSIBLES ────────────────────────────────────────────────────────────────────────────── */
document.querySelectorAll('.panel-title.collapsible').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById(btn.dataset.target).classList.toggle('collapsed');
    btn.classList.toggle('open');
  });
});

/* ── GENERATE ────────────────────────────────────────────────────────────────────────────── */
function updateGenerateBtn() {
  const disabled = !state.place || getSelectedModels().length === 0;
  document.getElementById('generate-btn').disabled = disabled;
  document.getElementById('excel-btn').disabled = disabled;
}

document.getElementById('generate-btn').addEventListener('click', async () => {
  const models = getSelectedModels();
  if (!state.place || !models.length) return;

  const payload = {
    lat:        state.place.lat,
    lon:        state.place.lon,
    name:       state.place.name,
    models,
    date_start: document.getElementById('date-start').value,
    date_end:   document.getElementById('date-end').value,
    wu_key:     document.getElementById('wu-key').value.trim(),
    stations:   [...state.selectedStations].map(id => {
      const s = state.stations.find(st => st.stationId === id);
      return { id, label: (s?.name || id).split(',')[0].trim() };
    }),
  };

  _modalPayload = { ...payload };
  _modalBaseRange = { date_start: payload.date_start, date_end: payload.date_end };
  openModal(state.place.name, payload.date_start, payload.date_end, models);

  try {
    const res = await fetch('/api/meteogram', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      showModalError(err.error || 'Unknown error');
      return;
    }
    const blob = await res.blob();
    showModalImage(URL.createObjectURL(blob));
  } catch (e) {
    showModalError(e.message);
  }
});

/* ── MODAL ────────────────────────────────────────────────────────────────────────────── */
function openModal(name, start, end, models) {
  document.getElementById('modal-title').textContent = `${name} — ${start} → ${end}`;

  // Meteogram tab reset
  document.getElementById('modal-spinner').style.display = '';
  document.getElementById('meteogram-img').classList.add('hidden');
  document.getElementById('modal-error').classList.add('hidden');

  // Zoom reset
  document.getElementById('meteo-zoom').classList.add('hidden');

  // Skew-T tab reset
  skewtState.loaded   = false;
  skewtState.loading  = false;
  skewtState.times    = [];
  document.getElementById('skewt-spinner').classList.add('hidden');
  document.getElementById('skewt-main').classList.add('hidden');
  document.getElementById('skewt-error').classList.add('hidden');
  document.getElementById('skewt-time-row').classList.add('hidden');
  initSkewtModelSel(models || []);

  // Preload Skew-T data in background while user reads meteogram
  setTimeout(() => {
    const model = document.getElementById('skewt-model-sel').value;
    if (model && WEATHER_MODELS[model]) loadSkewt();
  }, 0);

  // Activate Meteogram tab by default
  document.querySelectorAll('.modal-tab').forEach(b => b.classList.remove('active'));
  document.querySelector('.modal-tab[data-tab="tab-meteo"]').classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
  document.getElementById('tab-meteo').classList.remove('hidden');

  document.getElementById('modal').classList.remove('hidden');
}

function showModalImage(url) {
  const img = document.getElementById('meteogram-img');
  img.onload = () => {
    document.getElementById('modal-spinner').style.display = 'none';
    if (_modalBaseRange) {
      initZoomSliders(_modalBaseRange.date_start, _modalBaseRange.date_end);
      document.getElementById('meteo-zoom').classList.remove('hidden');
    }
  };
  img.src = url;
  img.classList.remove('hidden');
}

function showModalError(msg) {
  document.getElementById('modal-error').textContent = `Error: ${msg}`;
  document.getElementById('modal-error').classList.remove('hidden');
  document.getElementById('modal-spinner').style.display = 'none';
}

document.getElementById('excel-btn').addEventListener('click', async () => {
  const models = getSelectedModels();
  if (!state.place || !models.length) return;

  const payload = {
    lat:        state.place.lat,
    lon:        state.place.lon,
    name:       state.place.name,
    models,
    date_start: document.getElementById('date-start').value,
    date_end:   document.getElementById('date-end').value,
    wu_key:     document.getElementById('wu-key').value.trim(),
    stations:   [...state.selectedStations].map(id => {
      const s = state.stations.find(st => st.stationId === id);
      return { id, label: (s?.name || id).split(',')[0].trim() };
    }),
  };

  const btn = document.getElementById('excel-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Preparing…';

  try {
    const res = await fetch('/api/excel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      alert('Excel error: ' + (err.error || 'Unknown error'));
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `meteogram_${state.place.name}_${payload.date_start}.xlsx`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Excel error: ' + e.message);
  } finally {
    btn.disabled = !state.place || getSelectedModels().length === 0;
    btn.textContent = '⬇ Download Excel';
  }
});

document.getElementById('modal-close').addEventListener('click', () =>
  document.getElementById('modal').classList.add('hidden'));

document.getElementById('modal-backdrop').addEventListener('click', () =>
  document.getElementById('modal').classList.add('hidden'));

/* ── MODAL TABS ────────────────────────────────────────────────────────────────────────────── */
document.querySelectorAll('.modal-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.modal-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById(btn.dataset.tab).classList.remove('hidden');
    if (btn.dataset.tab === 'tab-skewt' && !skewtState.loaded && !skewtState.loading) loadSkewt();
  });
});

/* ── SKEW-T ────────────────────────────────────────────────────────────────────────────── */
const skewtState = { loaded: false, loading: false, times: [] };

function initSkewtModelSel(models) {
  const sel = document.getElementById('skewt-model-sel');
  const eligible = models.filter(m =>
    WEATHER_MODELS[m] && ['forecast', 'archive'].includes(WEATHER_MODELS[m].type));
  sel.innerHTML = eligible.length
    ? eligible.map(m => `<option value="${m}">${m}</option>`).join('')
    : '<option value="">— no eligible models —</option>';
}

document.getElementById('skewt-model-sel').addEventListener('change', () => {
  skewtState.loaded = false;
  loadSkewt();
});

document.getElementById('skewt-hour-sel').addEventListener('change', function () {
  fetchSkewtTime(this.value);
  updateNavArrows();
});

document.getElementById('skewt-prev').addEventListener('click', () => {
  const sel = document.getElementById('skewt-hour-sel');
  if (sel.selectedIndex > 0) {
    sel.selectedIndex--;
    fetchSkewtTime(sel.value);
    updateNavArrows();
  }
});

document.getElementById('skewt-next').addEventListener('click', () => {
  const sel = document.getElementById('skewt-hour-sel');
  if (sel.selectedIndex < sel.options.length - 1) {
    sel.selectedIndex++;
    fetchSkewtTime(sel.value);
    updateNavArrows();
  }
});

function updateNavArrows() {
  const sel = document.getElementById('skewt-hour-sel');
  document.getElementById('skewt-prev').disabled = sel.selectedIndex <= 0;
  document.getElementById('skewt-next').disabled = sel.selectedIndex >= sel.options.length - 1;
}

async function loadSkewt() {
  if (skewtState.loading) return;
  const model = document.getElementById('skewt-model-sel').value;
  if (!state.place || !model || !WEATHER_MODELS[model]) {
    showSkewtError('Select a valid forecast or archive model.');
    return;
  }
  skewtState.loading = true;
  document.getElementById('skewt-spinner').classList.remove('hidden');
  document.getElementById('skewt-main').classList.add('hidden');
  document.getElementById('skewt-error').classList.add('hidden');

  const result = await doFetchSkewt({ model, time: null });
  skewtState.loading = false;
  if (!result) return;

  skewtState.times  = result.times;
  skewtState.loaded = true;

  const hourSel = document.getElementById('skewt-hour-sel');
  hourSel.innerHTML = result.times.map(t =>
    `<option value="${t}"${t === result.time ? ' selected' : ''}>${t}</option>`
  ).join('');
  document.getElementById('skewt-time-row').classList.remove('hidden');
  updateNavArrows();
  showSkewtResult(result);
}

async function fetchSkewtTime(time) {
  const model = document.getElementById('skewt-model-sel').value;
  if (!model) return;
  document.getElementById('skewt-spinner').classList.remove('hidden');
  document.getElementById('skewt-main').classList.add('hidden');
  document.getElementById('skewt-error').classList.add('hidden');
  const result = await doFetchSkewt({ model, time });
  if (result) showSkewtResult(result);
}

async function doFetchSkewt({ model, time }) {
  const payload = {
    lat:        state.place.lat,
    lon:        state.place.lon,
    name:       state.place.name,
    model,
    date_start: document.getElementById('date-start').value,
    date_end:   document.getElementById('date-end').value,
  };
  if (time) payload.time = time;
  try {
    const res = await fetch('/api/skewt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    document.getElementById('skewt-spinner').classList.add('hidden');
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      showSkewtError(err.error || 'Unknown error');
      return null;
    }
    return await res.json();
  } catch (e) {
    document.getElementById('skewt-spinner').classList.add('hidden');
    showSkewtError(e.message);
    return null;
  }
}

function showSkewtResult(result) {
  document.getElementById('skewt-img').src = result.image;
  renderSkewtIndices(result.indices || {});
  document.getElementById('skewt-main').classList.remove('hidden');
}

function renderSkewtIndices(idx) {
  const rows = [
    { label: 'CAPE',      value: idx.cape         != null ? idx.cape.toFixed(0)         : '—', unit: 'J/kg' },
    { label: 'CIN',       value: idx.cin          != null ? idx.cin.toFixed(0)          : '—', unit: 'J/kg' },
    { label: 'LCL',       value: idx.lcl_hpa      != null ? idx.lcl_hpa.toFixed(0)      : '—', unit: 'hPa'  },
    { label: 'LCL T',     value: idx.lcl_temp     != null ? idx.lcl_temp.toFixed(1)     : '—', unit: '°C'   },
    { label: 'T disparo', value: idx.trigger_temp != null ? idx.trigger_temp.toFixed(1) : '—', unit: idx.trigger_temp != null ? '°C' : '' },
  ];
  document.getElementById('skewt-indices').innerHTML =
    '<div class="idx-title">Índices</div>' +
    rows.map(r => `<div class="idx-row">
      <span class="idx-label">${r.label}</span>
      <span class="idx-value">${escHtml(r.value)} <span class="idx-unit">${r.unit}</span></span>
    </div>`).join('');
}

function showSkewtError(msg) {
  document.getElementById('skewt-spinner').classList.add('hidden');
  document.getElementById('skewt-main').classList.add('hidden');
  const el = document.getElementById('skewt-error');
  el.textContent = `Error: ${msg}`;
  el.classList.remove('hidden');
}

/* ── ZOOM SLIDER ────────────────────────────────────────────────────────────────────────────── */
function initZoomSliders(dateStart, dateEnd) {
  const n = Math.round((new Date(dateEnd + 'T00:00:00') - new Date(dateStart + 'T00:00:00')) / 86400000);
  const startSl = document.getElementById('zoom-start-sl');
  const endSl   = document.getElementById('zoom-end-sl');
  startSl.min = 0; startSl.max = n; startSl.value = 0;
  endSl.min   = 0; endSl.max   = n; endSl.value   = n;
  updateZoomFill();
  updateZoomLabel();
}

function updateZoomFill() {
  const startSl = document.getElementById('zoom-start-sl');
  const endSl   = document.getElementById('zoom-end-sl');
  const n  = parseInt(startSl.max) || 1;
  const s  = parseInt(startSl.value) / n * 100;
  const e  = parseInt(endSl.value)   / n * 100;
  const fill = document.getElementById('dual-range-fill');
  fill.style.left  = s + '%';
  fill.style.width = (e - s) + '%';
}

function updateZoomLabel() {
  if (!_modalBaseRange) return;
  const startSl = document.getElementById('zoom-start-sl');
  const endSl   = document.getElementById('zoom-end-sl');
  const base = new Date(_modalBaseRange.date_start + 'T00:00:00');
  const d1 = new Date(base); d1.setDate(base.getDate() + parseInt(startSl.value));
  const d2 = new Date(base); d2.setDate(base.getDate() + parseInt(endSl.value));
  const fmt = d => d.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' });
  document.getElementById('zoom-label').textContent = `${fmt(d1)} – ${fmt(d2)}`;
}

document.getElementById('zoom-start-sl').addEventListener('input', function () {
  const endSl = document.getElementById('zoom-end-sl');
  if (parseInt(this.value) > parseInt(endSl.value) - 2) this.value = parseInt(endSl.value) - 2;
  updateZoomFill();
  updateZoomLabel();
});

document.getElementById('zoom-end-sl').addEventListener('input', function () {
  const startSl = document.getElementById('zoom-start-sl');
  if (parseInt(this.value) < parseInt(startSl.value) + 2) this.value = parseInt(startSl.value) + 2;
  updateZoomFill();
  updateZoomLabel();
});

document.getElementById('zoom-apply-btn').addEventListener('click', async () => {
  if (!_modalPayload || !_modalBaseRange) return;
  const startSl = document.getElementById('zoom-start-sl');
  const endSl   = document.getElementById('zoom-end-sl');
  const base     = new Date(_modalBaseRange.date_start + 'T00:00:00');
  const newStart = new Date(base); newStart.setDate(base.getDate() + parseInt(startSl.value));
  const newEnd   = new Date(base); newEnd.setDate(base.getDate() + parseInt(endSl.value));

  _modalPayload = { ..._modalPayload, date_start: fmtDate(newStart), date_end: fmtDate(newEnd) };

  document.getElementById('modal-spinner').style.display = '';
  document.getElementById('meteogram-img').classList.add('hidden');
  document.getElementById('meteo-zoom').classList.add('hidden');
  document.getElementById('modal-error').classList.add('hidden');

  try {
    const res = await fetch('/api/meteogram', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_modalPayload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      showModalError(err.error || 'Unknown error');
      return;
    }
    const blob = await res.blob();
    showModalImage(URL.createObjectURL(blob));
  } catch (e) {
    showModalError(e.message);
  }
});

/* ── UTILS ────────────────────────────────────────────────────────────────────────────── */
function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── INIT ────────────────────────────────────────────────────────────────────────────── */
initModelSelects();
initDates();

fetch('/api/config')
  .then(r => r.json())
  .then(cfg => { if (cfg.wu_api_key) document.getElementById('wu-key').value = cfg.wu_api_key; })
  .catch(() => {});
