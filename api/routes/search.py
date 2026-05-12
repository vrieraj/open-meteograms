import requests
from flask import Blueprint, request, jsonify

bp = Blueprint('search', __name__, url_prefix='/api')


@bp.route('/search')
def search():
    q = request.args.get('q', '').strip()
    if len(q) < 3:
        return jsonify([])
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'format': 'geocodejson', 'q': q, 'limit': 5,
                    'addressdetails': 1, 'namedetails': 1,
                    'accept-language': 'en-US,en;q=0.9'},
            headers={'User-Agent': 'OpenMeteograms/1.0'},
            timeout=5,
        )
        results = []
        for f in r.json().get('features', []):
            gc = f['properties']['geocoding']
            results.append({
                'label': gc.get('label', ''),
                'lat': f['geometry']['coordinates'][1],
                'lon': f['geometry']['coordinates'][0],
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/reverse')
def reverse():
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
    except (KeyError, ValueError):
        return jsonify({'error': 'lat and lon required'}), 400
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'format': 'geocodejson', 'lat': lat, 'lon': lon},
            headers={'User-Agent': 'OpenMeteograms/1.0'},
            timeout=5,
        )
        feats = r.json().get('features', [])
        if not feats:
            return jsonify({'error': 'not found'}), 404
        gc = feats[0]['properties']['geocoding']
        return jsonify({'label': gc.get('label', ''), 'lat': lat, 'lon': lon})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/place')
def place():
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
        name = request.args.get('name', '')
    except (KeyError, ValueError):
        return jsonify({'error': 'lat and lon required'}), 400
    try:
        from scripts.place import Place
        feature = {
            'geometry': {'coordinates': [lon, lat]},
            'properties': {'geocoding': {
                'label': name, 'name': name,
                'city': None, 'state': None,
                'country': None, 'country_code': None,
                'admin': {}, 'type': None,
            }}
        }
        p = Place(feature)
        sr, ss = None, None
        try:
            from astral import LocationInfo
            from astral.sun import sun as astral_sun
            from datetime import date as _date
            import pytz
            loc = LocationInfo(latitude=p.lat, longitude=p.lon,
                               timezone=str(p.tzinfo))
            s = astral_sun(loc.observer, date=_date.today(),
                           tzinfo=pytz.timezone(str(p.tzinfo)))
            sr = s['sunrise'].strftime('%H:%M') + ' h'
            ss = s['sunset'].strftime('%H:%M') + ' h'
        except Exception:
            pass
        return jsonify({
            'name': p.name,
            'lat': p.lat,
            'lon': p.lon,
            'elev': p.elev,
            'tz': str(p.tzinfo),
            'delta_time': p.delta_time,
            'urls': p.meteo,
            'map': p.map,
            'sunrise': sr,
            'sunset': ss,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
