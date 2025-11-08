from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from pre_soak_dataset import build_dataset, get_firms_data  # your data fusion script

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call API

# --- Helper function to convert NumPy types to native Python types ---
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# --- Simulation function ---
def run_your_simulation(lat=None, lon=None):
    # Build dataset for this location
    df = build_dataset(lat, lon)
    
    if df.empty:
        # Fallback values
        data = {
            "latitude": lat or 0.0,
            "longitude": lon or 0.0,
            "air_quality_index": 100,
            "pm2_5": 0,
            "pm10": 0,
            "ET": 0,
            "fire_intensity": 0,
            "nearest_fire_km": 20,
            "fires": []
        }
    else:
        row = df.iloc[0]
        # Fetch individual fires for map
        fires_df = get_firms_data(lat, lon)
        fires_list = []
        if not fires_df.empty:
            for _, f in fires_df.iterrows():
                fires_list.append({
                    "lat": f.get("latitude", lat),
                    "lon": f.get("longitude", lon),
                    "intensity": "High" if f.get("fire_intensity", 0) > 50 else "Medium"
                })

        data = {
            "latitude": lat,
            "longitude": lon,
            "air_quality_index": row.get("air_quality_index", 100),
            "pm2_5": row.get("pm2_5", 0),
            "pm10": row.get("pm10", 0),
            "ET": row.get("ET", 0),
            "fire_intensity": row.get("fire_intensity", 0),
            "nearest_fire_km": row.get("nearest_fire_km", 20),
            "fires": fires_list
        }
    return convert_numpy_types(data)

# --- API endpoint for simulation ---
@app.route('/api/run_simulation', methods=['GET'])
def api_run_simulation():
    lat = request.args.get('lat', type=float, default=36.77)
    lon = request.args.get('lon', type=float, default=-119.41)
    data = run_your_simulation(lat, lon)
    return jsonify(data)

# --- API endpoint for fire markers only ---
@app.route('/api/get_fires', methods=['GET'])
def api_get_fires():
    lat = request.args.get('lat', type=float, default=36.77)
    lon = request.args.get('lon', type=float, default=-119.41)
    df = get_firms_data(lat, lon)
    fires = []
    if not df.empty:
        for _, row in df.iterrows():
            fires.append({
                "lat": row.get("latitude", lat),
                "lon": row.get("longitude", lon),
                "intensity": "High" if row.get("fire_intensity", 0) > 50 else "Medium"
            })
    return jsonify(fires)

# --- Serve frontend ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)