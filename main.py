from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import math
import numpy as np
from pre_soak_dataset import build_dataset, get_firms_data  # your data fusion script

app = Flask(__name__)
CORS(app)

# --- Helper to convert NumPy types to native Python ---
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
DANGER_RADIUS_KM = 15

def is_within_radius(fire_lat, fire_lon, farm_lat, farm_lon, radius_km=DANGER_RADIUS_KM):
    lat_diff = fire_lat - farm_lat
    lon_diff = fire_lon - farm_lon
    km = math.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * math.cos(math.radians(farm_lat)))**2)
    return km <= radius_km

# --- Simulation function ---
def run_your_simulation(lat=None, lon=None, perimeter=50, pump_capacity=4250, demo=True):
    df = build_dataset(lat, lon)
    
    recommended_pump = perimeter * 85  # L/hr based on perimeter

    data = {
        "latitude": lat,
        "longitude": lon,
        "air_quality_index": df.get("air_quality_index", [100])[0],
        "pm2_5": df.get("pm2_5", [0])[0],
        "pm10": df.get("pm10", [0])[0],
        "ET": df.get("et_mm", [0])[0],
        "fire_intensity": df.get("fire_intensity", [0])[0],
        "nearest_fire_km": None,
        "fires": [],
        "perimeter": perimeter,
        "pump_capacity": pump_capacity,
        "recommended_pump": recommended_pump
    }

    # --- Fires ---
    fires_list = []
    if demo:
        mock_fire = {"lat": lat + 0.07, "lon": lon + 0.07, "intensity": "High"}
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

    # --- Risk scoring and recommendation ---
    fire_distance = data["nearest_fire_km"] if data["nearest_fire_km"] is not None else 20
    aqi = data["air_quality_index"]
    et_deficit = 1 - (data["ET"]/10) if data["ET"] is not None else 0

    risk_score = 0
    if fire_distance < 5: risk_score += 3
    elif fire_distance < 15: risk_score += 2
    if aqi > 200: risk_score += 3
    elif aqi > 100: risk_score += 2
    if et_deficit > 0.7: risk_score += 2
    if pump_capacity < recommended_pump: risk_score += 1

    # Recommendation logic
    if risk_score >= 7:
        recommendation = "CRITICAL RISK – FULL DEFENSE SOAK"
        rec_class = "bg-red-500 text-white border-red-700"
    elif risk_score >= 4:
        if pump_capacity < recommended_pump:
            recommendation = "MODERATE RISK – INCREASE PUMP CAPACITY & PERIMETER BOOST"
            rec_class = "bg-yellow-400 text-gray-900 border-yellow-500"
        else:
            recommendation = "MODERATE RISK – PERIMETER BOOST"
            rec_class = "bg-yellow-300 text-gray-900 border-yellow-500"
    else:
        recommendation = "NORMAL OPERATION – STANDARD IRRIGATION"
        rec_class = "bg-green-300 text-gray-900 border-green-500"

    data["ai_recommendation"] = recommendation
    data["ai_class"] = rec_class

    return convert_numpy_types(data)

# --- API endpoint ---
@app.route('/api/run_simulation', methods=['GET'])
def api_run_simulation():
    lat = request.args.get('lat', type=float, default=36.77)
    lon = request.args.get('lon', type=float, default=-119.41)
    perimeter = request.args.get('perimeter', type=float, default=50)
    pump_capacity = request.args.get('pump', type=float, default=4250)
    demo = request.args.get('demo', type=str, default="true").lower() == "true"

    data = run_your_simulation(lat, lon, perimeter, pump_capacity, demo)
    return jsonify(data)

# --- Serve frontend ---
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)