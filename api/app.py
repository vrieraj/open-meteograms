import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, send_from_directory, jsonify
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, '.env'))

STATIC = os.path.join(ROOT, 'static')
app = Flask(__name__, static_folder=STATIC, static_url_path='/static')

from api.routes.search import bp as search_bp
from api.routes.meteogram import bp as meteogram_bp
from api.routes.stations import bp as stations_bp

app.register_blueprint(search_bp)
app.register_blueprint(meteogram_bp)
app.register_blueprint(stations_bp)


@app.route('/')
def index():
    return send_from_directory(STATIC, 'index.html')


@app.route('/api/config')
def config():
    return jsonify({'wu_api_key': os.getenv('WU_API_KEY', '')})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)