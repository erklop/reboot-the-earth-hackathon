"""
Pre-Soak Prediction Data Fusion Script
Combines NASA FIRMS + OpenWeather + OpenET + Air Quality APIs
into one CSV dataset for AI model training.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from io import StringIO

# ========================
# CONFIGURATION
# ========================
LAT = 37.5         # Example: Central California nut orchard
LON = -120.8
RADIUS_DEG = 0.5   # Approx. 50 km search box for fires
DAYS_BACK = 7 # Max days to look back for OpenET and FIRMS

# API keys / tokens
load_dotenv()
FIRMS_KEY = os.getenv("FIRMS_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
OPENET_KEY = os.getenv("OPENET_KEY")  # Bearer token

# Output file
OUTPUT_CSV = "pre_soak_dataset.csv"

# Timezone
PST = ZoneInfo("America/Los_Angeles")

# Helper function
def safe_read_csv_from_response(resp, caller_name=""):
    """
    Tries to parse CSV from a response object.
    Returns empty DataFrame if parsing fails.
    """
    try:
        text = resp.content.decode("utf-8")
        df = pd.read_csv(StringIO(text))
        return df
    except Exception as e:
        print(f"{caller_name} — failed to parse CSV: {e}")
        snippet = getattr(resp, "text", "<no text>")[:500]
        print("Response snippet:", snippet)
        return pd.DataFrame()


# ========================
# NASA FIRMS - Fire Data
#
# FEATURES: fire_intensity, nearest_fire_km, num_fires_past_days
# This uses VIIRS_SNPP_NRT by default, but can use a diff dataset if applicable
# By default we look into the past 7 days of fires
# ========================
def get_firms_data(lat, lon, radius=0.5, days=DAYS_BACK, source="VIIRS_SNPP_NRT"):
    min_lon, min_lat = lon - radius, lat - radius
    max_lon, max_lat = lon + radius, lat + radius

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/{source}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}"

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = safe_read_csv_from_response(resp, "FIRMS")
        if df.empty:
            return pd.DataFrame([{"fire_intensity": 0, "nearest_fire_km": None, "num_fires_past_days": 0}])
        
        # compute summary metrics blahblahablah
        df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
        df["distance_km"] = (((df["latitude"] - lat) ** 2 + ((df["longitude"] - lon) * np.cos(np.radians(lat))) ** 2) ** 0.5) * 111
        return pd.DataFrame([{
            "fire_intensity": float(df["frp"].mean()),
            "nearest_fire_km": float(df["distance_km"].min()),
            "num_fires_past_days": len(df)
        }])
    # for debugging obv
    except Exception as e:
        print("FIRMS error:", e)
        return pd.DataFrame([{"fire_intensity": 0, "nearest_fire_km": None, "num_fires_past_days": 0}])


# ========================
# OpenWeather - Weather Data
#
# FEATURES: humidity, wind_speed, wind_deg
# Tis gives us a JSON we need to cherrypick data for the csv fr
# Checks rain in the past hour, everything else near real time
# ========================
def get_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_KEY}"

    try:
        # error in 15 seconsb hrbieaslushh
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        current = r.json().get("current", {})

        if not current:
            # if theres NOTHING bruh but pleaaaasse ain no way this dont work
            return pd.DataFrame()
        row = {
            "temperature": current.get("temp"),
            "humidity": current.get("humidity"),
            "wind_speed": current.get("wind_speed"),
            "wind_deg": current.get("wind_deg"),
            "rain_1h": current.get("rain").get("1h", 0) if isinstance(current.get("rain"), dict) else 0

        }
        return pd.DataFrame([row])
    except Exception as e:
        print("OpenWeather Onecall error:", e)
        return pd.DataFrame()

# ========================
# OpenWeather - Air Quality 
#
# AQI is what we need its the key feature we need AQI its the quality of the air the index of air quality
# I HATE JSON FILES GIVE ME CSV
# ========================
def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        results = r.json().get("list", [])

        # theres no way its empty bro
        if not results:
            return pd.DataFrame()
        
        components = results[0].get("components",{})
        aqi = results[0]["main"]["aqi"]

        return pd.DataFrame([{
            "air_quality_index": aqi,
            "pm2_5": components.get("pm2_5",0),
            "pm10": components.get("pm10",0)
        }])
    except Exception as e:
        print("OpenWeather Air Pollution error:", e)
        return pd.DataFrame()


# ========================
# OpenET - Evapotranspiration / Crop Stress
# Since OpenET API calls fetch a json file, we need to convert it to csv
# before merging into the final csv dataset.
# 
# Data from past 7 days real time yurrrrr
# Also we using PST timezone by default because we in Cali i do NOT care about timezone input rn
# ========================
def get_openet_data(lat, lon, days_back=DAYS_BACK):

    # WE IN CALIFORNIAAAAAAAAAAAAAAAAAAAAAA
    end_date = datetime.now(PST).date()
    start_date = end_date - timedelta(days=days_back)

    headers = {"Authorization": OPENET_KEY}
    payload = {
        "date_range": [str(start_date), str(end_date)],
        "interval": "daily",
        "geometry": [lon, lat],
        "model": "Ensemble",
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "file_format": "CSV"
    }

    try:
        resp = requests.post(
            url="https://openet-api.org/raster/timeseries/point",
            headers=headers,    # API KEY IS USED HERE DONT WORRY
            json=payload,
            timeout=30
        )
        resp.raise_for_status()

        # i hope this works
        df = safe_read_csv_from_response(resp, "OpenET")
        if df.empty:
            return pd.DataFrame()
        
        df["lat"], df["lon"] = lat, lon
        if "ET (mm)" in df.columns:
            df["et_mm"] = df["ET (mm)"]
        else:
            df["et_mm"] = 0

        df["et_cumulative_mm"] = df["et_mm"].cumsum()
        df["et_rolling_mean_3d"] = df["et_mm"].rolling(3).mean()
        df["et_rolling_mean_7d"] = df["et_mm"].rolling(7).mean()
        return df[["et_mm","et_cumulative_mm","et_rolling_mean_3d","et_rolling_mean_7d","lat","lon"]]
    except Exception as e:
        # please god no
        print("OpenET error:", e)
        return pd.DataFrame()

# ======================
# Compute Smoke Exposure Risk Index (SERI) for almond orchards
#
# Fire Threat (frp) weight: 25%
# Proximity (nearest_fire_km): 20%
# Air Quality (aqi): 20%
# PM2.5: 15%
# ET stress: 3 day rolling et: 20%
#
# ======================
def compute_SERI(df):
    # Safely extract scalar features
    fire_intensity = df["fire_intensity"].iloc[0] if "fire_intensity" in df else 0
    nearest_km = df["nearest_fire_km"].iloc[0] if "nearest_fire_km" in df and pd.notnull(df["nearest_fire_km"].iloc[0]) else 1000
    aqi = df["air_quality_index"].iloc[0] if "air_quality_index" in df else 1
    pm25 = df["pm2_5"].iloc[0] if "pm2_5" in df else 0
    et = df["et_rolling_mean_3d"].iloc[0] if "et_rolling_mean_3d" in df else 0

    # Fire Threat (FRP)
    fire_score = min(fire_intensity / 500 * 100, 100)  # scale FRP to 0-100

    # Fire Proximity
    distance_score = max(0, min(100, 100 - (nearest_km / 50 * 100)))  # closer = higher risk

    # Air Quality
    aqi_score = (aqi - 1)/4 * 100  # normalize 0-100

    # PM2.5
    pm25_score = min(pm25 / 150 * 100, 100)

    # ET stress (almonds need moist soil)
    et_score = max(0, 100 - min(et / 10 * 100, 100))

    # Weighted SERI
    seri = (
        fire_score * 0.25 +
        distance_score * 0.20 +
        aqi_score * 0.20 +
        pm25_score * 0.15 +
        et_score * 0.20
    )
    seri = np.clip(seri, 0, 100)

    # Assign Band
    if seri < 25:
        band = "LOW"
    elif seri < 50:
        band = "MODERATE"
    elif seri < 75:
        band = "HIGH"
    else:
        band = "CRITICAL"

    # Add to DataFrame
    df["SERI"] = seri
    df["SERI_band"] = band

    return df

# ========================
# Combine all datasets into one
# ========================
def build_dataset(lat=LAT, lon=LON, days_back=DAYS_BACK):
    print("Fetching FIRMS data...")
    firms_df = get_firms_data(lat, lon, days=days_back)

    print("Fetching weather data...")
    weather_df = get_weather_data(lat, lon)

    print("Fetching air quality data...")
    air_df = get_air_quality(lat, lon)

    print("Fetching OpenET data...")
    openet_df = get_openet_data(lat, lon, days_back=days_back)

    # Aggregate OpenET to single row (latest day)
    if not openet_df.empty:
        openet_agg = pd.DataFrame([{
            "et_mm": openet_df["et_mm"].iloc[-1],
            "et_rolling_mean_3d": openet_df["et_rolling_mean_3d"].iloc[-1],
            "et_rolling_mean_7d": openet_df["et_rolling_mean_7d"].iloc[-1],
        }])
    else:
        openet_agg = pd.DataFrame([{"et_mm": 0, "et_rolling_mean_3d": 0, "et_rolling_mean_7d": 0}])

    # Merge all single-row dataframes safely
    combined = pd.concat([firms_df.reset_index(drop=True),
                          weather_df.reset_index(drop=True),
                          air_df.reset_index(drop=True),
                          openet_agg.reset_index(drop=True)], axis=1)

    combined["lat"] = lat
    combined["lon"] = lon
    combined["date"] = datetime.now(PST).date()

    # Compute SERI
    combined = compute_SERI(combined)
    
    print("Dataset built successfully.")
    return combined


# ========================
# Save to CSV
# ========================
if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset saved to {OUTPUT_CSV}")
    print(df.head())