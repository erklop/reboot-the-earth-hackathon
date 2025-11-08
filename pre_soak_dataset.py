"""
Pre-Soak Prediction Data Fusion Script
Combines NASA FIRMS + OpenWeather + OpenET + Air Quality APIs
into one CSV dataset for AI model training.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

# ========================
# CONFIGURATION
# ========================
LAT = 37.5         # Example: Central California nut orchard
LON = -120.8
RADIUS_DEG = 0.5   # Approx. 50 km search box for fires

# API keys / tokens
FIRMS_KEY = "7719511bb3848d4b7c2d5a681b104d70"
OPENWEATHER_KEY = "16909f2013995623a841ba3ee73f48bb"
OPENET_TOKEN = "YyXnB2Mc5cCEUw37OGYM07VyUsaafUEcVPiZsDzRjwxVk0DimXebBVPbBTLX"  # Bearer token

# Output file
OUTPUT_CSV = "pre_soak_dataset.csv"


# ========================
# 1️NASA FIRMS - Fire Data
# ========================
def get_firms_data(lat, lon, radius=0.5, days="7d"):
    bbox = f"{lon-radius},{lat-radius},{lon+radius},{lat+radius}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT?bbox={bbox}&time={days}&api_key={FIRMS_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        print("FIRMS API error:", response.text)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        return pd.DataFrame([{"fire_intensity": 0, "nearest_fire_km": None}])

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["fire_intensity"] = df["frp"]
    df["distance"] = ((df["latitude"] - lat)**2 + (df["longitude"] - lon)**2)**0.5 * 111  # km approx
    nearest = df["distance"].min()
    avg_frp = df["fire_intensity"].mean()

    return pd.DataFrame([{
        "fire_intensity": avg_frp,
        "nearest_fire_km": nearest
    }])


# ========================
# OpenWeather - Weather Data
# ========================
def get_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    r = requests.get(url)
    data = r.json().get("current", {})
    if not data:
        return pd.DataFrame()

    return pd.DataFrame([{
        "temperature": data.get("temp"),
        "humidity": data.get("humidity"),
        "wind_speed": data.get("wind_speed"),
        "wind_deg": data.get("wind_deg"),
    }])


# ========================
# OpenET - Evapotranspiration / Crop Stress
# ========================
def get_openet_data(lat, lon, start_date=None, end_date=None):
    if not start_date:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=7)

    url = (
        f"https://openet-api.openetdata.org/timeseries?"
        f"lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&variable=et"
    )
    headers = {"Authorization": f"Bearer {OPENET_TOKEN}"}
    r = requests.get(url, headers=headers)
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["et"] = pd.to_numeric(df["et"], errors="coerce")
    et_cumulative = df["et"].sum()

    return pd.DataFrame([{"ET_cumulative": et_cumulative}])


# ========================
# Air Quality (OpenWeather)
# ========================
def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    r = requests.get(url)
    results = r.json().get("list", [])
    if not results:
        return pd.DataFrame()

    aqi = results[0]["main"]["aqi"]
    components = results[0]["components"]
    return pd.DataFrame([{
        "air_quality_index": aqi,
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
    }])


# ========================
# 5️Combine all
# ========================
def build_dataset(lat=LAT, lon=LON):
    print("Fetching data...")

    firms_df = get_firms_data(lat, lon)
    weather_df = get_weather_data(lat, lon)
    openet_df = get_openet_data(lat, lon)
    air_df = get_air_quality(lat, lon)

    combined = pd.concat([firms_df, weather_df, openet_df, air_df], axis=1)
    combined["lat"] = lat
    combined["lon"] = lon
    combined["date"] = datetime.utcnow().date()

    print("Data fetched successfully.")
    return combined


# ========================
# Save to CSV
# ========================
if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset saved to {OUTPUT_CSV}")
    print(df.head())