import ee
import os
import logging
import pandas as pd
import numpy as np
from rasterio import open as rio_open
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool, freeze_support, cpu_count
import sys

# ---------------- SUPPRESS EE LOGS ---------------- #
logging.getLogger('earthengine').setLevel(logging.ERROR)

class DevNull:
    def write(self, msg): pass
    def flush(self): pass

sys.stdout = DevNull()
ee.Initialize(project = "weather-470911")
sys.stdout = sys.__stdout__

# ---------------- CONFIG ---------------- #
DEM_PATH = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_clipped_dem.tif"
LANDCOVER_PATH = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_landcover_2020.tif"
OUTPUT_DIR = r"C:\Users\gfeka\Desktop\phyton"

# Manual polygon definitions
polygons_dict = {
    "Dire_Dawa_Center": ee.Geometry.Polygon([[[41.75, 9.55], [41.95, 9.55], [41.95, 9.65], [41.75, 9.65]]]),
    "Dire_Dawa_East": ee.Geometry.Polygon([[[42.00, 9.50], [42.20, 9.50], [42.20, 9.65], [42.00, 9.65]]])
}

CN_LOOKUP = {10: 55, 20: 70, 30: 85, 40: 61, 50: 77, 60: 100}

logging.basicConfig(level=logging.INFO)

# ---------------- HELPER FUNCTIONS ---------------- #
def compute_polygon_cn(polygon, landcover_path=LANDCOVER_PATH):
    with rio_open(landcover_path) as src:
        geom = [mapping(polygon)]
        out_image, _ = mask(src, geom, crop=True)
        data = out_image[0]
        vals = data[data != src.nodata]
        if len(vals) == 0:
            return 75
        cn_vals = [CN_LOOKUP.get(v, 75) for v in vals]
        return float(np.mean(cn_vals))

def compute_polygon_slope(dem_path, polygon):
    with rio_open(dem_path) as src:
        geom = [mapping(polygon)]
        out_image, _ = mask(src, geom, crop=True)
        dem_data = out_image[0].astype(float)
        dem_data[dem_data == src.nodata] = np.nan
        grad_y, grad_x = np.gradient(dem_data)
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_percent = np.nanmean(np.tan(slope_rad) * 100)
        return slope_percent

def estimate_discharge_cn_slope(row):
    """Combined CN + slope discharge estimation for a single polygon row."""
    rainfall_mm, polygon = row["rain_mm"], row["geometry"]
    CN = compute_polygon_cn(polygon)
    if rainfall_mm <= 0:
        return 0.0
    S = (25400.0 / CN) - 254.0
    Ia = 0.2 * S
    if rainfall_mm <= Ia:
        Q = 0.0
    else:
        Q = ((rainfall_mm - Ia) ** 2) / (rainfall_mm - Ia + S)
    slope_percent = compute_polygon_slope(DEM_PATH, polygon)
    Q_adjusted = Q * (1 + 0.02 * slope_percent)
    return Q_adjusted

def fetch_rainfall(item):
    """Fetch mean annual rainfall from CHIRPS for a polygon."""
    try:
        name, ee_geom = item
        shapely_geom = Polygon(ee_geom.getInfo()['coordinates'][0])
        ee_geom_poly = ee.Geometry.Polygon(list(shapely_geom.exterior.coords))
        dataset = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate("2024-01-01", "2024-12-31") \
            .filterBounds(ee_geom_poly)
        rain = dataset.select("precipitation").mean().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom_poly,
            scale=5000,
            maxPixels=1e13
        )
        rain_mm = rain.getInfo().get("precipitation", 0.0)
        return {"polygon_name": name, "rain_mm": rain_mm, "geometry": shapely_geom}
    except Exception as e:
        logging.error(f"Error fetching rainfall for {name}: {e}")
        return {"polygon_name": name, "rain_mm": 0.0, "geometry": shapely_geom}

def train_rf_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ---------------- MAIN PIPELINE ---------------- #
def main():
    freeze_support()  # Needed for Windows
    logging.info("🚀 Starting fully parallel rainfall + discharge pipeline...")

    # Determine optimal number of CPU cores
    num_processes = min(len(polygons_dict), cpu_count())

    # Step 1: Parallel rainfall fetching
    with Pool(processes=num_processes) as pool:
        records = pool.map(fetch_rainfall, polygons_dict.items())
    rainfall_df = pd.DataFrame(records)

    # Step 2: Parallel slope + CN discharge estimation
    with Pool(processes=num_processes) as pool:
        discharges = pool.map(estimate_discharge_cn_slope, [row for _, row in rainfall_df.iterrows()])
    rainfall_df["discharge_estimated"] = discharges

    # Step 3: Random Forest prediction
    X = rainfall_df[["rain_mm"]]
    y = rainfall_df["discharge_estimated"]
    rf_model = train_rf_model(X, y)
    rainfall_df["discharge_predicted"] = rf_model.predict(X)

    # Step 4: Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "river_volume_prediction.csv")
    rainfall_df.drop(columns="geometry").to_csv(output_file, index=False)

    logging.info(f"✅ River volume prediction saved to {output_file}")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()
