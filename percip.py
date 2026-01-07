import os
import sys
import time
import logging
import datetime as dt
import requests
import numpy as np
import pandas as pd
import rasterio
import richdem as rd
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from collections import deque
import joblib
import threading

# ------------------------------- CONFIGURATION -------------------------------
CITY = "Dire Dawa"
LAT, LON = 9.6000, 41.8661
API_KEY = "cd8abc61a64e458abcf121843252908"
INTERVAL = 300  # seconds (5 minutes)
HIST_WINDOW = 12
PROJECT_DIR = r"C:\Users\gfeka\Desktop\phyton"
DEM_PATH = os.path.join(PROJECT_DIR, "dire_dawa_clipped_dem.tif")
LC_PATH = os.path.join(PROJECT_DIR, "dire_dawa_landcover_2020.tif")
CSV_FILE = os.path.join(PROJECT_DIR, "dire_dawa_flood_log.csv")
os.makedirs(PROJECT_DIR, exist_ok=True)

# --------------------------- TELEGRAM CONFIGURATION ---------------------------
TELEGRAM_BOT_TOKEN = "8263513287:AAFb9yu02M1v7hb96C9SDFMpaSvm83GQUFo"
TELEGRAM_CHAT_ID = "QWESDFASWEDSBOT"

# ------------------------------- LOGGING -------------------------------
logging.basicConfig(
    filename=os.path.join(PROJECT_DIR, "flood_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------- HISTORICAL WEATHER -------------------------------
weather_history = deque(maxlen=HIST_WINDOW)
history_lock = threading.Lock()

# ------------------------------- TELEGRAM ALERT FUNCTION -------------------------------
def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        logging.info("Telegram alert sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

# ------------------------------- WEATHER FETCHER THREAD -------------------------------
def weather_updater():
    """Runs forever in background, fetching new weather data every INTERVAL seconds"""
    while True:
        url_current = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LAT},{LON}"
        url_forecast = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={LAT},{LON}&hours=6"
        try:
            r_current = requests.get(url_current, timeout=10)
            r_current.raise_for_status()
            current = r_current.json()
            current_precip = float(current["current"]["precip_mm"])

            r_forecast = requests.get(url_forecast, timeout=10)
            r_forecast.raise_for_status()
            forecast = r_forecast.json()
            forecast_precip = sum(h.get("precip_mm", 0.0) for h in forecast["forecast"]["forecastday"][0]["hour"][:6])

            weather = {
                "datetime": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_precip": current_precip,
                "forecast_precip": forecast_precip,
                "temperature": current["current"]["temp_c"],
                "humidity": current["current"]["humidity"],
                "wind_kph": current["current"]["wind_kph"]
            }

            with history_lock:
                weather_history.append(weather)

            logging.info(f"Weather updated: {weather}")

        except Exception as e:
            logging.error(f"Weather API error: {e}")

        time.sleep(INTERVAL)  # wait before next fetch

# ------------------------------- RASTER HELPERS -------------------------------
def load_and_resample(base_path, target_shape, target_transform):
    with rasterio.open(base_path) as src:
        arr = src.read(1)
        resampled = np.empty(target_shape, dtype=np.float32)
        reproject(
            source=arr,
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )
        return resampled

def normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    mn, mx = arr[finite].min(), arr[finite].max()
    return (arr - mn) / (mx - mn + 1e-6)

def create_features(dem, slope, flow_acc, lc_term):
    with history_lock:
        H = len(weather_history)
        h_precip = [w["current_precip"] for w in weather_history]
        h_forecast = [w["forecast_precip"] for w in weather_history]
        h_temp = [w["temperature"] for w in weather_history]
        h_hum = [w["humidity"] for w in weather_history]
        h_wind = [w["wind_kph"] for w in weather_history]

    for arr in [h_precip, h_forecast, h_temp, h_hum, h_wind]:
        while len(arr) < HIST_WINDOW:
            arr.insert(0, 0.0)

    features = [
        dem.flatten(),
        slope.flatten(),
        flow_acc.flatten(),
        lc_term.flatten()
    ]

    for h in range(HIST_WINDOW):
        features.extend([
            np.full(dem.size, h_precip[h]),
            np.full(dem.size, h_forecast[h]),
            np.full(dem.size, h_temp[h]),
            np.full(dem.size, h_hum[h]),
            np.full(dem.size, h_wind[h])
        ])

    return np.stack(features, axis=1)

def save_tif(path, array, like_ds, nodata=-9999.0):
    profile = like_ds.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        arr = np.array(array, dtype=np.float32)
        arr[np.isnan(arr)] = nodata
        dst.write(arr, 1)

# ------------------------------- PRECOMPUTE STATIC -------------------------------
def precompute_static():
    if not (os.path.exists("dem.npy") and os.path.exists("slope.npy")):
        dem_ds = rasterio.open(DEM_PATH)
        dem = rd.LoadGDAL(DEM_PATH)
        slope = rd.TerrainAttribute(dem, attrib="slope_degrees")
        flow_acc = rd.FlowAccumulation(dem, method="D8")
        lc_term = load_and_resample(LC_PATH, dem.shape, dem_ds.transform)
        np.save("dem.npy", dem)
        np.save("slope.npy", slope)
        np.save("flow.npy", flow_acc)
        np.save("lc.npy", lc_term)
    dem = np.load("dem.npy")
    slope = np.load("slope.npy")
    flow_acc = np.load("flow.npy")
    lc_term = np.load("lc.npy")
    dem_ds = rasterio.open(DEM_PATH)
    return dem, slope, flow_acc, lc_term, dem_ds

# ------------------------------- TRAIN MODELS ONCE -------------------------------
def train_models(dem, slope, flow_acc, lc_term):
    if os.path.exists("sm_model.pkl") and os.path.exists("rl_model.pkl"):
        sm_model = joblib.load("sm_model.pkl")
        rl_model = joblib.load("rl_model.pkl")
    else:
        X_dummy = create_features(dem, slope, flow_acc, lc_term)
        y_dummy_sm = 0.2 + 0.05 * np.random.randn(X_dummy.shape[0])
        y_dummy_rl = 0.1 + 0.05 * np.random.randn(X_dummy.shape[0])

        sm_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        sm_model.fit(X_dummy, y_dummy_sm)

        rl_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        rl_model.fit(np.c_[X_dummy, y_dummy_sm], y_dummy_rl)

        joblib.dump(sm_model, "sm_model.pkl")
        joblib.dump(rl_model, "rl_model.pkl")
    return sm_model, rl_model

# ------------------------------- MAIN LOOP -------------------------------
def run_flood_detection():
    dem, slope, flow_acc, lc_term, dem_ds = precompute_static()
    sm_model, rl_model = train_models(dem, slope, flow_acc, lc_term)

    alert_threshold = 0.7

    while True:
        with history_lock:
            if not weather_history:
                print("Waiting for first weather update...")
                time.sleep(10)
                continue
            latest = weather_history[-1]

        X_features = create_features(dem, slope, flow_acc, lc_term)
        sm_pred = sm_model.predict(X_features).reshape(dem.shape)
        X_rl_features = np.c_[X_features, sm_pred.flatten()]
        river_pred = rl_model.predict(X_rl_features).reshape(dem.shape)

        dem_n = normalize(dem)
        slope_n = normalize(slope)
        flow_n = normalize(flow_acc)
        lc_n = normalize(lc_term)
        sm_n = normalize(sm_pred)
        river_n = normalize(river_pred)

        precip_n = normalize(np.full_like(dem, latest["current_precip"]))
        forecast_n = normalize(np.full_like(dem, latest["forecast_precip"]))

        flood_index = (
            0.1 * dem_n + 0.1 * lc_n +
            0.3 * precip_n + 0.3 * forecast_n +
            0.1 * sm_n + 0.1 * river_n
        )

        df = pd.DataFrame({
            "datetime": [latest["datetime"]],
            "current_precip": [latest["current_precip"]],
            "forecast_precip": [latest["forecast_precip"]],
            "flood_index_mean": [float(np.mean(flood_index))],
            "soil_moisture_mean": [float(np.mean(sm_pred))],
            "river_level_mean": [float(np.mean(river_pred))],
            "temperature": [latest["temperature"]],
            "humidity": [latest["humidity"]],
            "wind_kph": [latest["wind_kph"]]
        })
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode="a", index=False, header=False)

        max_index = np.max(flood_index)
        if max_index > alert_threshold:
            alert_msg = (f"⚠️ Flood Alert! Max index = {max_index:.2f}\n"
                         f"Location: {CITY}\n"
                         f"Time: {latest['datetime']}\n"
                         f"Current precip: {latest['current_precip']} mm\n"
                         f"Forecast precip: {latest['forecast_precip']} mm")
            print(alert_msg)
            logging.info(alert_msg)
            send_telegram_alert(alert_msg)
            save_tif(os.path.join(PROJECT_DIR, "flood_index.tif"), flood_index, dem_ds)
        else:
            print(f"No immediate flood risk. Max index = {max_index:.2f}")

        time.sleep(INTERVAL)

# ------------------------------- ENTRY POINT -------------------------------
if __name__ == "__main__":
    try:
        # Start weather thread
        threading.Thread(target=weather_updater, daemon=True).start()
        run_flood_detection()
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        logging.info("Flood detection stopped manually.")
