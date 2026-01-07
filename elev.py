import richdem as rd
import matplotlib.pyplot as plt
import requests
import os
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import numpy as np
import time
import datetime as dt

# -------------------------------
# Project settings
# -------------------------------
project_dir = r"C:\Users\gfeka\Desktop\phyton"
os.makedirs(project_dir, exist_ok=True)
os.chdir(project_dir)

API_KEY = "7a7d64ed416b5126dd5f6059a8ef3811"

tile_width = 0.25
tile_height = 0.25

merged_dem_file = "dire_dawa_merged_dem.tif"
clipped_dem_file = "dire_dawa_clipped_dem.tif"

# -------------------------------
# Define polygon/basin boundary
# -------------------------------
polygon_coords = [
    (41.6, 9.4),
    (42.1, 9.4),
    (42.1, 9.8),
    (41.6, 9.8),
    (41.6, 9.4)
]

poly_shape = Polygon(polygon_coords)
polygon = gpd.GeoDataFrame({'geometry': [poly_shape]}, crs="EPSG:4326")

# -------------------------------
# Logging function
# -------------------------------
def log(msg):
    now = dt.datetime.now(dt.timezone.utc)
    print(f"[{now.isoformat()}] {msg}")

# -------------------------------
# Helper functions
# -------------------------------
def download_dem_tile(tile_name, west, south, east, north, retries=3):
    dem_file = f"{tile_name}.tif"
    if not os.path.exists(dem_file):
        for attempt in range(retries):
            try:
                log(f"Downloading DEM tile: {tile_name}")
                url = (
                    "https://portal.opentopography.org/API/globaldem"
                    "?demtype=SRTMGL1"
                    f"&west={west}&south={south}&east={east}&north={north}"
                    "&outputFormat=GTiff"
                    f"&API_Key={API_KEY}"
                )
                r = requests.get(url, stream=True, timeout=30)
                if r.status_code == 200:
                    with open(dem_file, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            f.write(chunk)
                    log(f"Tile {tile_name} downloaded successfully!")
                    break
                else:
                    log(f"HTTP {r.status_code} error, retrying...")
            except Exception as e:
                log(f"Error: {e}, retrying...")
            time.sleep(2)
        else:
            raise Exception(f"Failed to download {tile_name} after {retries} attempts.")
    else:
        log(f"Tile {tile_name} already exists. Skipping download.")
    return ensure_crs(dem_file)

def ensure_crs(dem_file, crs="EPSG:4326"):
    with rasterio.open(dem_file) as src:
        if src.crs is None:
            log(f"CRS missing in {dem_file}, creating fixed GeoTIFF with {crs}")
            meta = src.meta.copy()
            meta.update({"crs": crs})
            temp_file = dem_file.replace(".tif", "_fixed.tif")
            with rasterio.open(temp_file, "w", **meta) as dst:
                dst.write(src.read())
            return temp_file
    return dem_file

def generate_tiles(bbox, tile_width, tile_height):
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    idx = 1
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            tiles.append((f"tile{idx}", lon, lat,
                          min(lon + tile_width, lon_max),
                          min(lat + tile_height, lat_max)))
            lon += tile_width
            idx += 1
        lat += tile_height
    return tiles

# -------------------------------
# Generate tiles and download DEM
# -------------------------------
minx, miny, maxx, maxy = polygon.total_bounds
bbox = [minx, miny, maxx, maxy]
tiles = generate_tiles(bbox, tile_width, tile_height)
dem_files = [download_dem_tile(*tile) for tile in tiles]

# -------------------------------
# Merge DEM tiles
# -------------------------------
src_files_to_mosaic = [rasterio.open(f) for f in dem_files]
mosaic, out_trans = merge(src_files_to_mosaic)
crs = src_files_to_mosaic[0].crs or "EPSG:4326"

with rasterio.open(
    merged_dem_file, "w", driver="GTiff",
    height=mosaic.shape[1], width=mosaic.shape[2], count=1,
    dtype=mosaic.dtype, crs=crs, transform=out_trans, nodata=-9999
) as dest:
    dest.write(mosaic[0], 1)

for src in src_files_to_mosaic:
    src.close()

log(f"Merged DEM saved as {merged_dem_file}")

# -------------------------------
# Clip DEM to polygon
# -------------------------------
with rasterio.open(merged_dem_file) as src:
    if polygon.crs != src.crs:
        polygon = polygon.to_crs(src.crs)
    out_image, out_transform = mask(src, [mapping(polygon.geometry[0])], crop=True, nodata=-9999)
    out_meta = src.meta.copy()
    out_meta.update({"height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     "nodata": -9999})

with rasterio.open(clipped_dem_file, "w", **out_meta) as dest:
    dest.write(out_image)

log(f"Clipped DEM saved as {clipped_dem_file}")

# -------------------------------
# Load DEM into richdem
# -------------------------------
dem = rd.LoadGDAL(clipped_dem_file)
slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
aspect = rd.TerrainAttribute(dem, attrib='aspect')
flow_acc = rd.FlowAccumulation(dem, method='D8')

# -------------------------------
# Plot all four maps in a 2x2 grid
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

maps = [
    (dem, 'terrain', "Clipped DEM Elevation", "Elevation (m)"),
    (slope, 'inferno', "Slope Map", "Slope (degrees)"),
    (aspect, 'twilight', "Aspect Map", "Aspect (degrees)"),
    (flow_acc, 'Blues', "Flow Accumulation", "Flow Accumulation")
]

for ax, (data, cmap, title, cbar_label) in zip(axes.flatten(), maps):
    im = ax.imshow(data, cmap=cmap, origin='upper')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.2, hspace=0.25)

# -------------------------------
# Save figure as high-resolution PNG
# -------------------------------
png_file = "dire_dawa_all_maps.png"
plt.savefig(png_file, dpi=600, bbox_inches='tight')
log(f"Saved combined figure as PNG: {png_file}")

# -------------------------------
# Optional: Save as GeoTIFF (DEM only, as GIS can't store 2x2 figure directly)
# -------------------------------
with rasterio.open(clipped_dem_file) as src:
    meta = src.meta.copy()
    meta.update(dtype=rasterio.float32, count=1)

tiff_file = "dire_dawa_dem_for_gis.tif"
with rasterio.open(tiff_file, 'w', **meta) as dst:
    dst.write(dem.astype(rasterio.float32), 1)

log(f"Saved DEM GeoTIFF for GIS: {tiff_file}")

plt.show()
