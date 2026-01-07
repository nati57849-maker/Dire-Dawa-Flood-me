import os
import logging
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, sobel
from datetime import datetime, timedelta
from numba import njit, prange
from tqdm import tqdm
import imageio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ------------------- CONFIG -------------------
API_KEY = "cd8abc61a64e458abcf121843252908"
CITY = "Dire Dawa"
DEM_PATH = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_clipped_dem.tif"
LANDCOVER_PATH = r"C:\Users\gfeka\Desktop\phyton\landcover_aligned.tif"
SOIL_PATH = r"C:\Users\gfeka\Desktop\phyton\soil_aligned.tif"
RIVER_PATH = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_rivers.tif"
OUTPUT_DIR = r"C:\Users\gfeka\Desktop\phyton\soil_moisture_maps_full"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "soil_moisture_stack.tif")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "soil_moisture_stats.csv")
OUTPUT_GIF = os.path.join(OUTPUT_DIR, "soil_moisture_animation.gif")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DAYS_BACK = 14
END_DATE = datetime.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=DAYS_BACK - 1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- SOIL PARAMETERS -------------------
def soil_fraction_to_params(clay_frac):
    clay = np.clip(clay_frac, 0.0, 1.0)
    FC = 70.0 + 200.0 * clay
    WP = 15.0 + 100.0 * clay
    k = np.maximum(0.005, 0.25 - 0.22 * clay)
    return FC, WP, k

# ------------------- CURVE NUMBER LOOKUP -------------------
LC_TO_CN = {
    11: 98,  # water
    21: 75,  # urban low
    22: 85,  # urban med
    23: 90,  # urban high
    31: 77,  # barren
    41: 55,  # forest
    42: 60,  # shrub
    71: 68,  # grassland
    81: 65,  # pasture
    82: 72,  # cropland
}

# ------------------- WEATHER + ET -------------------
session = requests.Session()

def compute_FAO_PM(T_mean, Tmin, Tmax, RH_mean, wind, Rs):
    G = 0
    z = 1260
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = 0.000665 * P
    es_Tmax = 0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3))
    es_Tmin = 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))
    es = (es_Tmax + es_Tmin) / 2
    ea = es * RH_mean / 100
    delta = 4098 * (0.6108 * np.exp((17.27 * T_mean) / (T_mean + 237.3))) / (T_mean + 237.3) ** 2
    Rn = 0.77 * Rs
    ET0 = (0.408 * delta * (Rn - G) + gamma * (900 / (T_mean + 273)) * wind * (es - ea)) / (
            delta + gamma * (1 + 0.34 * wind))
    return max(0, ET0)

def fetch_weather_for_date(date, city=CITY, api_key=API_KEY, retries=3):
    dt_str = date.strftime("%Y-%m-%d")
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={dt_str}"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=12)
            r.raise_for_status()
            day = r.json()['forecast']['forecastday'][0]['day']
            T_mean = day['avgtemp_c']
            Tmin = day['mintemp_c']
            Tmax = day['maxtemp_c']
            RH_mean = day['avghumidity']
            wind = day['maxwind_kph'] / 3.6
            solar_rad = day.get('daily_chance_of_sunshine', 60) / 100.0 * 20
            total_rain = sum(float(h.get('precip_mm', 0) or 0) for h in r.json()['forecast']['forecastday'][0]['hour'])
            et0 = compute_FAO_PM(T_mean, Tmin, Tmax, RH_mean, wind, solar_rad)
            return {'date': date.date(), 'rainfall_mm': total_rain, 'ET0_mm': et0}
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {dt_str}: {e}")
            if attempt == retries - 1:
                return {'date': date.date(), 'rainfall_mm': 0.0, 'ET0_mm': 2.0}

def fetch_daily_weather_with_ET_parallel(start_date, end_date, cache_path=os.path.join(OUTPUT_DIR, "weather_cache.csv")):
    if os.path.exists(cache_path):
        logging.info("Using cached weather data")
        return pd.read_csv(cache_path, parse_dates=['date'])
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_weather_for_date, d) for d in dates]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda x: x['date'])
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    return df

# ------------------- UTILITIES -------------------
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        mask = np.ones(arr.shape, dtype=bool)
        if nodata is not None:
            mask = arr != nodata
        return arr, mask, src.transform, src.crs

def compute_river_proximity_factor(river_arr):
    binary_river = (river_arr > 0).astype(int)
    distance = distance_transform_edt(1 - binary_river)
    max_dist = np.max(distance)
    proximity_factor = 1 - (distance / (max_dist + 1e-6))
    return proximity_factor

def compute_slope_factor(dem_arr):
    dx = sobel(dem_arr, axis=1)
    dy = sobel(dem_arr, axis=0)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    slope_norm = slope / (np.max(slope) + 1e-6)
    return slope_norm

# ------------------- BUCKET MODEL -------------------
@njit(parallel=True)
def run_bucket_model_multilayer_step(S_top, S_sub, P_day, ET_day,
                                     FC_top, WP_top, FC_sub, WP_sub,
                                     k_top, k_sub, CN_arr, gw_depth,
                                     slope_factor, river_factor):
    n_pixels = FC_top.size
    new_top = np.zeros(n_pixels, dtype=np.float32)
    new_sub = np.zeros(n_pixels, dtype=np.float32)
    for idx in prange(n_pixels):
        P = P_day[idx]
        ET = ET_day[idx]
        CN = CN_arr[idx]
        slope = slope_factor[idx]
        river = river_factor[idx]

        CN_adj = min(100, CN + 15 * slope)
        S = (25400.0 / CN_adj) - 254.0
        Ia = 0.2 * S
        runoff = 0.0
        if P > Ia:
            runoff = ((P - Ia) ** 2) / (P - Ia + S)
        infil = max(P - runoff, 0.0)

        S_top[idx] += infil
        S_top[idx] -= 0.7 * ET

        if S_top[idx] > FC_top[idx]:
            excess = S_top[idx] - FC_top[idx]
            S_top[idx] = FC_top[idx]
            S_sub[idx] += excess

        S_sub[idx] -= 0.3 * ET
        drain = k_sub[idx] * max(0.0, S_sub[idx] - FC_sub[idx])
        S_sub[idx] -= drain

        cap_rise = k_sub[idx] * np.exp(-0.5 * gw_depth[idx]) * (1 + 0.5 * river)
        S_sub[idx] += cap_rise

        if S_top[idx] < FC_top[idx]:
            delta = min(cap_rise * 0.3, FC_top[idx] - S_top[idx])
            S_top[idx] += delta
            S_sub[idx] -= delta

        S_top[idx] = min(max(S_top[idx], WP_top[idx]), FC_top[idx])
        S_sub[idx] = min(max(S_sub[idx], WP_sub[idx]), FC_sub[idx])
        new_top[idx] = S_top[idx]
        new_sub[idx] = S_sub[idx]
    return new_top, new_sub

# ------------------- BLOCK PROCESS -------------------
def process_block_slice(i, j, block_size, dem_arr, lc_arr, soil_arr, river_arr, slope_arr, rain_day, ET_day):
    block_slice = (slice(i, i+block_size), slice(j, j+block_size))
    dem_block = dem_arr[block_slice].astype(np.float32)
    lc_block = lc_arr[block_slice].astype(np.int32)
    soil_block = soil_arr[block_slice].astype(np.float32)

    valid_mask = (~np.isnan(dem_block)) & (~np.isnan(lc_block)) & (~np.isnan(soil_block))
    if not valid_mask.any():
        return i, j, np.full(dem_block.shape, -9999, dtype=np.float32)

    valid_idx = np.where(valid_mask.flatten())[0]
    clay_flat = soil_block.flatten()[valid_idx]
    FC, WP, k = soil_fraction_to_params(clay_flat)
    FC_top, WP_top, k_top = FC * 0.5, WP * 0.5, k
    FC_sub, WP_sub, k_sub = FC * 0.8, WP * 0.8, k * 0.5

    lc_flat = lc_block.flatten()[valid_idx]
    CN_arr = np.array([LC_TO_CN.get(int(lc), 70) for lc in lc_flat])
    slope_flat = slope_arr.flatten()[valid_idx]
    river_flat = river_arr.flatten()[valid_idx]

    n_valid = valid_idx.size
    rain_day_block = np.full(n_valid, rain_day)
    ET_day_block = np.full(n_valid, ET_day)
    gw_depth = np.full(n_valid, 5.0)
    S_top_init = np.zeros(n_valid, dtype=np.float32)
    S_sub_init = np.zeros(n_valid, dtype=np.float32)

    S_top_new, S_sub_new = run_bucket_model_multilayer_step(
        S_top_init, S_sub_init, rain_day_block, ET_day_block,
        FC_top, WP_top, FC_sub, WP_sub,
        k_top, k_sub, CN_arr, gw_depth, slope_flat, river_flat)

    sm_block = np.full(dem_block.size, -9999, dtype=np.float32)
    sm_block[valid_idx] = S_top_new
    return i, j, sm_block.reshape(dem_block.shape)

# ------------------- MAIN SIMULATION -------------------
def simulate_soil_moisture(block_size=512, gif_step=2):
    logging.info("Fetching weather data...")
    weather_df = fetch_daily_weather_with_ET_parallel(START_DATE, END_DATE)
    rain_series = weather_df['rainfall_mm'].values
    et_series = weather_df['ET0_mm'].values
    dates = weather_df['date'].values
    n_days = len(dates)

    river_arr, _, _, _ = read_raster(RIVER_PATH)
    river_factor = compute_river_proximity_factor(river_arr)
    dem_arr, _, transform, crs = read_raster(DEM_PATH)
    slope_factor = compute_slope_factor(dem_arr)
    lc_arr, _, _, _ = read_raster(LANDCOVER_PATH)
    soil_arr, _, _, _ = read_raster(SOIL_PATH)

    n_rows, n_cols = dem_arr.shape
    blocks = [(i, j) for i in range(0, n_rows, block_size) for j in range(0, n_cols, block_size)]

    gif_frames = []
    stats = []

    with rasterio.open(
        OUTPUT_FILE, 'w',
        driver='GTiff',
        height=n_rows,
        width=n_cols,
        count=n_days,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:

        for b, date in enumerate(tqdm(dates, desc="Simulating days")):
            sm_day = np.full((n_rows, n_cols), -9999, dtype=np.float32)

            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(process_block_slice, i, j, block_size,
                                           dem_arr, lc_arr, soil_arr, river_factor, slope_factor,
                                           rain_series[b], et_series[b]) for i, j in blocks]
                for future in as_completed(futures):
                    i, j, sm_block = future.result()
                    sm_day[i:i + sm_block.shape[0], j:j + sm_block.shape[1]] = sm_block

            dst.write(sm_day, b + 1)

            # Compute stats
            mask = sm_day != -9999
            vals = sm_day[mask]
            stats.append({
                'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p50': float(np.percentile(vals, 50)),
                'p75': float(np.percentile(vals, 75)),
            })

            # GIF frame
            if b % gif_step == 0 or b == n_days - 1:
                frame = (np.clip(sm_day, 0, 300) / 300 * 255).astype(np.uint8)
                gif_frames.append(frame)

    pd.DataFrame(stats).to_csv(OUTPUT_STATS, index=False)
    imageio.mimsave(OUTPUT_GIF, gif_frames, duration=0.8)

    logging.info(f"Simulation complete. GeoTIFF: {OUTPUT_FILE}, Stats: {OUTPUT_STATS}, GIF: {OUTPUT_GIF}")
    return OUTPUT_FILE, OUTPUT_STATS, OUTPUT_GIF

if __name__ == "__main__":
    simulate_soil_moisture()
