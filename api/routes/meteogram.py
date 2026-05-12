import io
import pandas as pd
from flask import Blueprint, request, jsonify, send_file

bp = Blueprint('meteogram', __name__, url_prefix='/api')

REQUIRED = ('lat', 'lon', 'name', 'models', 'date_start', 'date_end')
COL_ORDER = [
    'time', 'temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
    'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m',
    'vapour_pressure_deficit', 'fuel_moisture', 'fuel_moisture_vpd',
    'prob_ignition',
]


def _build_place_feature(data):
    return {
        'geometry': {'coordinates': [data['lon'], data['lat']]},
        'properties': {'geocoding': {
            'label': data['name'], 'name': data['name'],
            'city': data.get('city'), 'state': data.get('state_name'),
            'country': data.get('country'), 'country_code': None,
            'admin': {}, 'type': None,
        }}
    }


def _build_sfc(data):
    from scripts.place import Place
    from scripts.meteo_sfc import MeteoSfc

    place = Place(_build_place_feature(data))
    fechas = [data['date_start'], data['date_end']]
    sfc = MeteoSfc(place, fechas)
    sfc.get_data('openmeteo', models=data['models'])

    wu_key = data.get('wu_key', '').strip()
    for s in data.get('stations', []):
        if not wu_key:
            break
        from datasources.wx_stations import fetch_wu_hourly
        df = fetch_wu_hourly(s['id'], fechas[0], fechas[1], wu_key)
        if df is not None and not df.empty:
            sfc.get_data_station(df, label=s.get('label', s['id']))

    return place, sfc


@bp.route('/meteogram', methods=['POST'])
def meteogram():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return jsonify({'error': f'Missing: {", ".join(missing)}'}), 400

    try:
        from scripts.meteo_vrt import MeteoVrt
        from scripts.weather_models import WEATHER_MODELS

        place, sfc = _build_sfc(data)
        fechas = [data['date_start'], data['date_end']]
        models = data['models']

        vrt = None
        if (len(models) == 1
                and WEATHER_MODELS.get(models[0], {}).get('type') == 'forecast'):
            vrt = MeteoVrt(place, fechas)
            vrt.get_data('openmeteo', model=models[0])

        fig = sfc.meteoplot(vrt=vrt)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/excel', methods=['POST'])
def excel():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return jsonify({'error': f'Missing: {", ".join(missing)}'}), 400

    try:
        _, sfc = _build_sfc(data)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            for src in sfc.datos['source'].unique():
                label = sfc.station_names.get(src, src)
                sheet = label[:31]
                df_src = sfc.datos[sfc.datos['source'] == src].copy()
                cols = [c for c in COL_ORDER if c in df_src.columns]
                df_src[cols].to_excel(writer, sheet_name=sheet, index=False)
        buf.seek(0)

        filename = f"meteogram_{data['name']}_{data['date_start']}.xlsx"
        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename,
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/skewt', methods=['POST'])
def skewt():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    for k in ('lat', 'lon', 'name', 'model', 'date_start', 'date_end'):
        if k not in data:
            return jsonify({'error': f'Missing: {k}'}), 400

    try:
        import base64
        import matplotlib.pyplot as plt
        from scripts.place import Place
        from scripts.meteo_vrt import MeteoVrt
        from scripts.weather_models import WEATHER_MODELS

        model = data['model']
        if model not in WEATHER_MODELS:
            return jsonify({'error': f'Unknown model: {model}'}), 400

        place = Place(_build_place_feature(data))
        fechas = [data['date_start'], data['date_end']]
        time = data.get('time')

        vrt = MeteoVrt(place, fechas)
        vrt.get_data('openmeteo', model=model)
        vrt._fetch_skewt_data()

        if vrt._datos_skewt is None:
            return jsonify({'error': 'No vertical data available'}), 500

        times = vrt._datos_skewt['time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        if not times:
            return jsonify({'error': 'No time steps available'}), 500

        if time is None:
            noon = [t for t in times if t.endswith('12:00')]
            time = noon[0] if noon else times[0]

        indices = vrt.compute_skewt_indices(time)

        fig = vrt.skewt(time)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        return jsonify({
            'times': times,
            'time': time,
            'image': f'data:image/png;base64,{img_b64}',
            'indices': indices,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
