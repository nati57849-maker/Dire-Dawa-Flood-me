import rasterio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd

# --- Dire Dawa bounding box (approx) ---
min_lon, min_lat = 41.81, 9.55
max_lon, max_lat = 41.95, 9.65

# --- Load downloaded SoilGrids rasters ---
clay_raster = "clay_0-5cm.tif"
silt_raster = "silt_0-5cm.tif"
sand_raster = "sand_0-5cm.tif"

def extract_soil_texture(raster_path, bounds):
    with rasterio.open(raster_path) as src:
        # Get the window corresponding to the bounding box
        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform)
        data = src.read(1, window=window)  # read first band
        transform = src.window_transform(window)
    return data, transform

bounds = (min_lon, min_lat, max_lon, max_lat)

clay_data, clay_transform = extract_soil_texture(clay_raster, bounds)
silt_data, silt_transform = extract_soil_texture(silt_raster, bounds)
sand_data, sand_transform = extract_soil_texture(sand_raster, bounds)

# --- Flatten arrays and create a DataFrame ---
rows, cols = clay_data.shape
soil_texture = pd.DataFrame({
    "clay": clay_data.flatten(),
    "silt": silt_data.flatten(),
    "sand": sand_data.flatten()
})

# Remove no-data values if necessary (assuming SoilGrids uses 0 as no-data)
soil_texture = soil_texture[(soil_texture != 0).all(axis=1)]

# Save to CSV
soil_texture.to_csv("dire_dawa_soil_texture.csv", index=False)

print("Soil texture for Dire Dawa saved as 'dire_dawa_soil_texture.csv'")
