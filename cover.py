import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import richdem as rd
from rasterio.warp import reproject, Resampling

# -------------------------------
# Project Settings
# -------------------------------
PROJECT_DIR = r"C:\Users\gfeka\Desktop\phyton"
lc_file = os.path.join(PROJECT_DIR, "dire_dawa_landcover_2020.tif")
dem_file = os.path.join(PROJECT_DIR, "dire_dawa_dem.tif")

# -------------------------------
# Load Land Cover GeoTIFF
# -------------------------------
with rasterio.open(lc_file) as src:
    lc = src.read(1)
    lc_transform = src.transform
    lc_meta = src.meta.copy()

# -------------------------------
# Load DEM GeoTIFF
# -------------------------------
with rasterio.open(dem_file) as src:
    dem = src.read(1)
    dem_transform = src.transform
    dem_crs = src.crs

# -------------------------------
# Detect city extent (built-up = 50)
# -------------------------------
built_up_class = 50
coords = np.argwhere(lc == built_up_class)
if coords.size == 0:
    raise ValueError("No built-up pixels found in the land cover map.")

y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
buffer = 50
y_min = max(y_min - buffer, 0)
y_max = min(y_max + buffer, lc.shape[0])
x_min = max(x_min - buffer, 0)
x_max = min(x_max + buffer, lc.shape[1])

# Crop Land Cover
lc_crop = lc[y_min:y_max, x_min:x_max]

# -------------------------------
# Resample DEM to match Land Cover Crop
# -------------------------------
dem_resampled = np.empty_like(lc_crop, dtype=np.float32)

with rasterio.open(dem_file) as src:
    reproject(
        source=dem,
        destination=dem_resampled,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=lc_transform,
        dst_crs=lc_meta["crs"],
        resampling=Resampling.bilinear
    )

# -------------------------------
# Map land cover to runoff coefficient
# -------------------------------
runoff_coeff = np.zeros_like(lc_crop, dtype=float)
runoff_coeff[lc_crop == 50] = 0.9  # Built-up
runoff_coeff[lc_crop == 40] = 0.5  # Cropland
runoff_coeff[lc_crop == 30] = 0.3  # Grass
runoff_coeff[lc_crop == 10] = 0.2  # Tree
runoff_coeff[lc_crop == 80] = 1.0  # Water

# -------------------------------
# Land Cover colormap
# -------------------------------
lc_classes = [10, 30, 40, 50, 60, 80]
lc_labels = ['Tree (10)', 'Grass (30)', 'Cropland (40)', 'Built-up (50)', 'Bare (60)', 'Water (80)']
lc_colors = ['#006400', '#7CFC00', '#FFD700', '#FF0000', '#BEBEBE', '#0000FF']
cmap_lc = mcolors.ListedColormap(lc_colors)
norm_lc = mcolors.BoundaryNorm(boundaries=[0, 11, 31, 41, 51, 61, 81], ncolors=len(lc_colors))

# -------------------------------
# Compute hillshade
# -------------------------------
dem_rd = rd.rdarray(dem_resampled, no_data=-9999)
slope = rd.TerrainAttribute(dem_rd, attrib='slope_radians')
aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')

hs = 255.0 * ((np.cos(slope) * np.cos(np.radians(45))) +
              (np.sin(slope) * np.sin(np.radians(45)) *
               np.cos(aspect - np.radians(315))))
hs = np.clip(hs, 0, 255)

# -------------------------------
# Flood Index (normalized risk map)
# -------------------------------
slope_norm = (slope - np.nanmin(slope)) / (np.nanmax(slope) - np.nanmin(slope) + 1e-6)
base_runoff = runoff_coeff
flood_index = base_runoff * (1 + slope_norm)

# Normalize 0–1
flood_norm = (flood_index - np.nanmin(flood_index)) / (np.nanmax(flood_index) - np.nanmin(flood_index) + 1e-6)

# Classify risk
flood_risk = np.zeros_like(flood_norm, dtype=int)
flood_risk[flood_norm < 0.25] = 1
flood_risk[(flood_norm >= 0.25) & (flood_norm < 0.5)] = 2
flood_risk[(flood_norm >= 0.5) & (flood_norm < 0.75)] = 3
flood_risk[flood_norm >= 0.75] = 4

# Risk colormap
risk_colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000']
risk_labels = ['Low', 'Moderate', 'High', 'Very High']
cmap_risk = mcolors.ListedColormap(risk_colors)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm_risk = mcolors.BoundaryNorm(bounds, cmap_risk.N)

# -------------------------------
# Combined Plot (up-down layout)
# -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 18))

# --- Top: Land Cover ---
axes[0].imshow(hs, cmap='gray', alpha=1)
axes[0].imshow(lc_crop, cmap=cmap_lc, norm=norm_lc, alpha=0.7)
patches = [mpatches.Patch(color=lc_colors[i], label=lc_labels[i]) for i in range(len(lc_labels))]
axes[0].legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, framealpha=0.7, facecolor='white', fontsize=10)
axes[0].set_title("Dire Dawa City: Hillshade + Land Cover", fontsize=14)
axes[0].axis('off')

# --- Bottom: Flood Risk ---
axes[1].imshow(hs, cmap='gray', alpha=1)
axes[1].imshow(flood_risk, cmap=cmap_risk, norm=norm_risk, alpha=0.6)
patches = [mpatches.Patch(color=risk_colors[i], label=risk_labels[i]) for i in range(len(risk_labels))]
axes[1].legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=4, framealpha=0.7, facecolor='white', fontsize=10)
axes[1].set_title("Dire Dawa City: Flood Risk Levels", fontsize=14)
axes[1].axis('off')

# Add white space between maps
plt.subplots_adjust(hspace=0.4)

# Save figure
out_png = os.path.join(PROJECT_DIR, "landcover_floodrisk_up_down.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Maps generated and saved at: {out_png}")
