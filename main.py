from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from pre_soak_dataset import build_dataset  # your existing CSV builder

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
    # Try building dataset
    df = build_dataset(lat, lon)
    
    if df.empty:
        # Default/fallback values
        data = {
            "latitude": np.float64(lat if lat else 0.0),
            "longitude": np.float64(lon if lon else 0.0),
            "air_quality_index": np.int64(100),
            "pm2_5": np.float64(0),
            "pm10": np.float64(0),
            "ET": np.float64(0),
            "fire_intensity": np.float64(0),
            "nearest_fire_km": np.float64(20),
            "result_array": np.array([np.int64(1), np.int64(2), np.int64(3)]),
            "summary": {"max_val": np.int64(3), "mean_val": np.float64(2.0)}
        }
    else:
        row = df.iloc[0]
        data = {
            "latitude": np.float64(lat),
            "longitude": np.float64(lon),
            "air_quality_index": row.get("air_quality_index", 100),
            "pm2_5": row.get("pm2_5", 0),
            "pm10": row.get("pm10", 0),
            "ET": row.get("ET", 0),
            "fire_intensity": row.get("fire_intensity", 0),
            "nearest_fire_km": row.get("nearest_fire_km", 20),
            "result_array": np.array([np.int64(1), np.int64(2), np.int64(3)]),
            "summary": {"max_val": np.int64(3), "mean_val": np.float64(2.0)}
        }
    # Convert all NumPy types to native Python
    return convert_numpy_types(data)

# --- API endpoint ---
@app.route('/api/run_simulation', methods=['GET'])
def api_run_simulation():
    lat = request.args.get('lat', type=float, default=36.77)
    lon = request.args.get('lon', type=float, default=-119.41)
    data = run_your_simulation(lat, lon)
    return jsonify(data)

# --- Serve frontend ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)