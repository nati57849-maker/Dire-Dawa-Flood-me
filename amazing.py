import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
import rasterio
import richdem as rd
import datetime as dt

# -------------------------------
# CONFIGURATION
# -------------------------------
CITY = "Dire Dawa"
LAT, LON = 9.6000, 41.8661
API_KEY = "cd8abc61a64e458abcf121843252908"
INTERVAL = 3600  # seconds (1 hour)
CSV_FILE = "dire_dawa_weather.csv"
PROJECT_DIR = r"C:\Users\gfeka\Desktop\phyton"
DEM_PATH = os.path.join(PROJECT_DIR, "dire_dawa_clipped_dem.tif")
AOI_POINT = (41.86, 9.60)

os.makedirs(PROJECT_DIR, exist_ok=True)

# -------------------------------
# LOGGING SETUP
# -------------------------------
logging.basicConfig(
    filename=os.path.join(PROJECT_DIR, "weather_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# WEATHERAPI FUNCTIONS
# -------------------------------
def fetch_weather():
    """Fetch current weather for immediate flood risk."""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LAT},{LON}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather = {
            "city": CITY,
            "datetime": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_c": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "precip_mm": data["current"]["precip_mm"],
            "wind_kph": data["current"]["wind_kph"],
            "condition": data["current"]["condition"]["text"],
            "cloud": data["current"]["cloud"]
        }
        return weather
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        return None

def save_to_csv(weather):
    """Append weather data to CSV."""
    df = pd.DataFrame([weather])
    if not os.path.isfile(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode="a", index=False, header=False)

def fetch_hourly_forecast():
    """Fetch next-hour forecast for planning/logging purposes."""
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={LAT},{LON}&hours=24"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        hours = data["forecast"]["forecastday"][0]["hour"]
        forecast = [{"local_time": dt.datetime.fromisoformat(h["time"]),
                     "precip_mm": float(h.get("precip_mm",0.0))} for h in hours]
        next_hour_precip = forecast[0]["precip_mm"] if forecast else None
        return next_hour_precip
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching hourly forecast: {e}")
        return None

# -------------------------------
# DEM / TOPOGRAPHY FUNCTIONS
# -------------------------------
def ensure_topo_layers(dem_path):
    dem = rd.LoadGDAL(dem_path)
    slope = rd.TerrainAttribute(dem, attrib="slope_degrees")
    dem_int = rd.rdarray(dem, no_data=-9999, dtype=int)
    flow_acc = rd.FlowAccumulation(dem_int, method="D8")
    return dem, slope, flow_acc

def normalize(arr):
    a = np.array(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    mn, mx = a[finite].min(), a[finite].max()
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - mn) / (mx - mn)
    out[~finite] = 0
    return out

def flood_risk(flow_acc, slope_deg, rain_mm_per_hr):
    fa_n = normalize(flow_acc)
    sl_term = 1.0 / (1.0 + np.maximum(slope_deg, 0.0))
    rain_term = np.maximum(rain_mm_per_hr, 0.0)
    return fa_n * sl_term * rain_term

def save_tif(path, array, like_ds, nodata=-9999.0, dtype=rasterio.float32):
    profile = like_ds.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        arr = array.astype(dtype)
        arr[np.isnan(arr)] = nodata
        dst.write(arr,1)

# -------------------------------
# COUNTDOWN FUNCTION
# -------------------------------
def countdown(seconds):
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        sys.stdout.write(f"\rNext fetch in {mins:02d}:{secs:02d}...")
        sys.stdout.flush()
        time.sleep(1)
    print("\rFetching now!")

# -------------------------------
# MAIN LOOP
# -------------------------------
def start_collection():
    print(f"Starting hybrid weather & flood risk collection for {CITY}...\n")
    
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")
    dem_ds = rasterio.open(DEM_PATH)
    dem, slope, flow_acc = ensure_topo_layers(DEM_PATH)

    while True:
        # --- Current weather for flood risk ---
        weather = fetch_weather()
        if weather:
            print("Collected Weather:", weather)
            save_to_csv(weather)
            logging.info(f"Weather data collected: {weather}")
        else:
            print("Weather fetch failed. Retrying in 1 min...")
            countdown(60)
            continue
        
        # --- Next-hour forecast (optional logging) ---
        next_hour_fcst_mm = fetch_hourly_forecast()
        if next_hour_fcst_mm is not None:
            logging.info(f"Next-hour forecast precipitation: {next_hour_fcst_mm} mm")
        
        # --- Flood risk calculation ---
        rain_field = np.full((dem_ds.height, dem_ds.width), weather["precip_mm"], dtype=np.float32)
        risk = flood_risk(flow_acc, slope, rain_field)

        # --- Save outputs ---
        save_tif(os.path.join(PROJECT_DIR, "current_rain_mmhr.tif"), rain_field, dem_ds)
        save_tif(os.path.join(PROJECT_DIR, "flood_risk_index.tif"), risk, dem_ds)
        print("Saved current_rain_mmhr.tif and flood_risk_index.tif")

        countdown(INTERVAL)

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    try:
        start_collection()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting safely...")
        logging.info("Process stopped manually.")
