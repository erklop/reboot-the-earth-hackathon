from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from pre_soak_dataset import build_dataset, get_firms_data  # your data fusion script

app = Flask(__name__)
CORS(app)

# --- Helper to convert NumPy types ---
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

# --- Fire radius helper ---
DANGER_RADIUS_KM = 15  # Only consider fires within 15 km

def is_within_radius(fire_lat, fire_lon, farm_lat, farm_lon, radius_km=DANGER_RADIUS_KM):
    lat_diff = fire_lat - farm_lat
    lon_diff = fire_lon - farm_lon
    km = math.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * math.cos(math.radians(farm_lat)))**2)
    return km <= radius_km

# --- Simulation function ---
def run_your_simulation(lat=None, lon=None, perimeter=50, pump_capacity=4250, demo=True):
    df = build_dataset(lat, lon)
    
    # Recommended pump ~ 85 L/hr per ft of perimeter
    recommended_pump = perimeter * 85

    if df.empty:
        data = {
            "latitude": lat or 0.0,
            "longitude": lon or 0.0,
            "air_quality_index": 100,
            "pm2_5": 0,
            "pm10": 0,
            "ET": 0,
            "fire_intensity": 0,
            "nearest_fire_km": None,
            "fires": [],
            "perimeter": perimeter,
            "pump_capacity": pump_capacity,
            "recommended_pump": recommended_pump
        }
    else:
        row = df.iloc[0]
        data = {
            "latitude": lat,
            "longitude": lon,
            "air_quality_index": row.get("air_quality_index", 100),
            "pm2_5": row.get("pm2_5", 0),
            "pm10": row.get("pm10", 0),
            "ET": row.get("ET", 0),
            "fire_intensity": row.get("fire_intensity", 0),
            "nearest_fire_km": None,
            "fires": [],
            "perimeter": perimeter,
            "pump_capacity": pump_capacity,
            "recommended_pump": recommended_pump
        }

    fires_list = []

    if demo:
        mock_fire = {
            "lat": lat + 0.07,  # ~8 km north
            "lon": lon + 0.07,  # ~8 km east
            "intensity": "High"
        }
        fires_list.append(mock_fire)
    else:
        fires_df = get_firms_data(lat, lon)
        if not fires_df.empty:
            for _, f in fires_df.iterrows():
                if is_within_radius(f["latitude"], f["longitude"], lat, lon):
                    fires_list.append({
                        "lat": f["latitude"],
                        "lon": f["longitude"],
                        "intensity": "High" if f.get("fire_intensity", 0) > 50 else "Medium"
                    })

    data["fires"] = fires_list

    if fires_list:
        distances = [
            math.sqrt((f["lat"] - lat)**2 + (f["lon"] - lon)**2) * 111
            for f in fires_list
        ]
        data["nearest_fire_km"] = min(distances)

    return convert_numpy_types(data)

# --- API endpoint for simulation ---
@app.route('/api/run_simulation', methods=['GET'])
def api_run_simulation():
    lat = request.args.get('lat', type=float, default=37.6)
    lon = request.args.get('lon', type=float, default=-120.9)
    perimeter = request.args.get('perimeter', type=float, default=50)
    pump_capacity = request.args.get('pump', type=float, default=4250)
    demo = request.args.get('demo', type=str, default="true").lower() == "true"
    data = run_your_simulation(lat, lon, perimeter, pump_capacity, demo)
    return jsonify(data)

# --- API endpoint for fire markers only ---
@app.route('/api/get_fires', methods=['GET'])
def api_get_fires():
    lat = request.args.get('lat', type=float, default=37.6)
    lon = request.args.get('lon', type=float, default=-120.9)
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