"""
Pre-Soak Prediction Data Fusion Script
Combines NASA FIRMS + OpenWeather + OpenET + Air Quality APIs
into one CSV dataset for AI model training.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# API keys / tokens
# Retrieve API keys from environment
FIRMS_KEY = os.getenv("FIRMS_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
OPENET_KEY = os.getenv("OPENET_KEY")

if not all([FIRMS_KEY, OPENWEATHER_KEY, OPENET_KEY]):
    raise ValueError("One or more API keys are missing from .env!")

# ========================
# CONFIGURATION
# ========================
LAT = 37.5         # Example: Central California nut orchard
LON = -120.8
RADIUS_DEG = 0.5   # Approx. 50 km search box for fires


# Output file
OUTPUT_CSV = "pre_soak_dataset.csv"


# ========================
# NASA FIRMS - Fire Data
# This uses VIIRS_SNPP_NRT data specifically
# ========================
def get_firms_data(lat, lon, radius=0.5, days="7"):
    bbox = f"{lon-radius},{lat-radius},{lon+radius},{lat+radius}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/VIIRS_SNPP_NRT/{bbox}/{days}"

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
# Since OpenET API calls fetch a json file, we need to convert it to csv
# before merging into the final csv dataset.
# ========================
def get_openet_data(lat, lon, start_date=None, end_date=None):
    if end_date is None:
        end_date = datetime.utcnow().date()
    if start_date is None:
        start_date = end_date - timedelta(days=7)

    header = {"Authorization": f"Bearer {OPENET_KEY}"}
    args = {
        "date_range": [str(start_date), str(end_date)],
        "interval": "daily",
        "geometry": [lon, lat],
        "model": "Ensemble",
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "file_format": "CSV"
    }

    resp = requests.post(
        url="https://openet-api.org/raster/timeseries/point",
        headers=header,
        json=args
    )

    if resp.status_code != 200:
        print("OpenET request failed:", resp.text)
        return pd.DataFrame()

    # Save and load CSV directly
    csv_path = "openet_data.csv"
    with open(csv_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(csv_path)
    df["lat"] = lat
    df["lon"] = lon
    return df

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