from flask import Blueprint, request, jsonify

bp = Blueprint('stations', __name__, url_prefix='/api')


@bp.route('/stations')
def stations():
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
        radius_km = float(request.args.get('radius_km', 50))
        api_key = request.args.get('api_key', '').strip()
    except (KeyError, ValueError):
        return jsonify({'error': 'lat, lon and api_key required'}), 400

    if not api_key:
        return jsonify({'error': 'api_key required'}), 400

    try:
        from datasources.wx_stations import fetch_wu_stations_near
        result = fetch_wu_stations_near(lat, lon, radius_km, api_key)
        return jsonify(result or [])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
