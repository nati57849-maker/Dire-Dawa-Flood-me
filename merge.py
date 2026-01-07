# ====== Soil Moisture ML for Dire Dawa (GEE + RandomForest) ======
# Predict SMAP soil moisture using Sentinel-1 VV/VH + CHIRPS rainfall
# Window: last 180 days, 3-day aggregates

import ee
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------
# 0) Initialize Earth Engine
# ---------------------------
try:
    ee.Initialize(project='weather-470911')  # or ee.Initialize(project='your-project-id')
except Exception:
    ee.Authenticate()
    ee.Initialize()

# ---------------------------
# 1) AOI: Dire Dawa (~15 km buffer)
# ---------------------------
dire_dawa_pt = ee.Geometry.Point([41.8661, 9.6008])
aoi = dire_dawa_pt.buffer(15000)

# ---------------------------
# 2) Dates (auto): last 180 days
# ---------------------------
end_py = dt.date.today()
start_py = end_py - dt.timedelta(days=180)
start = ee.Date(start_py.strftime("%Y-%m-%d"))
end   = ee.Date(end_py.strftime("%Y-%m-%d"))

step_days = 3  # aggregation window

def build_dates(start_ee, end_ee, step):
    n = end_ee.difference(start_ee, 'day').divide(step).floor().int()
    return ee.List.sequence(0, n.subtract(1)).map(
        lambda i: start_ee.advance(ee.Number(i).multiply(step), 'day')
    )

date_list = build_dates(start, end, step_days)

# ---------------------------
# 3) Collections
# ---------------------------
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .select(['VV', 'VH']))

chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
          .filterBounds(aoi)
          .select(['precipitation']))

smap = (ee.ImageCollection('NASA/SMAP/SPL3SMP_E/006')
        .filterBounds(aoi)
        .select(['soil_moisture_am', 'soil_moisture_pm']))

# ---------------------------
# 4) Helper: reduce to AOI means per 3-day window
# ---------------------------
def window_feature(d):
    d = ee.Date(d)
    d2 = d.advance(step_days, 'day')

    s1_win = s1.filterDate(d, d2).mean()
    rain_win = chirps.filterDate(d, d2).sum()
    smap_win = smap.filterDate(d, d2) \
                   .map(lambda im: im.select(['soil_moisture_am','soil_moisture_pm'])
                                   .reduce(ee.Reducer.mean()).rename(['sm'])) \
                   .mean()

    s1_dict = s1_win.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi, scale=20, maxPixels=1e13
    )
    rain_dict = rain_win.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi, scale=5600, maxPixels=1e13
    )
    smap_dict = smap_win.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi, scale=9000, maxPixels=1e13
    )

    out = ee.Dictionary({'date': d.format('YYYY-MM-dd')}) \
        .combine(s1_dict, overwrite=True) \
        .combine(rain_dict, overwrite=True) \
        .combine(smap_dict, overwrite=True)

    return ee.Feature(None, out)

fc = ee.FeatureCollection(date_list.map(window_feature))

# ---------------------------
# 5) Bring data to Python
# ---------------------------
table = fc.getInfo()  # small (~60 rows), safe to bring client-side
rows = [f['properties'] for f in table['features']]
df = pd.DataFrame(rows)

# Clean up
df['date'] = pd.to_datetime(df['date'])
for col in ['VV', 'VH', 'precipitation', 'sm']:
    if col not in df:
        df[col] = np.nan

df = df.sort_values('date')
df = df.dropna(subset=['VV','VH','precipitation'], how='all').reset_index(drop=True)

print("Sample of training table:")
print(df.head())

# ---------------------------
# 6) Train ML model (RF)
# ---------------------------
train = df.dropna(subset=['sm'])
if len(train) < 10:
    raise RuntimeError(f"Not enough training samples with SMAP ({len(train)} rows).")

X = train[['VV','VH','precipitation']].fillna(method='ffill').fillna(method='bfill')
y = train['sm'].astype(float)

rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X, y)

# ---------------------------
# 7) Predict soil moisture
# ---------------------------
df['predicted_sm'] = rf.predict(
    df[['VV','VH','precipitation']].fillna(method='ffill').fillna(method='bfill')
)

# ---------------------------
# 8) Evaluate model accuracy
# ---------------------------
valid = df.dropna(subset=['sm'])
y_true = valid['sm']
y_pred = valid['predicted_sm']

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
accuracy_percent = r2 * 100

print("\n✅ Model Performance on Available SMAP Data:")
print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.4f} m³/m³")
print(f"Approximate accuracy: {accuracy_percent:.1f}%")

# ---------------------------
# 9) Save CSV
# ---------------------------
df.to_csv('dire_dawa_smap_ml_3day.csv', index=False)
print("\n✅ Saved: dire_dawa_smap_ml_3day.csv")

# ---------------------------
# 10) Plot predicted soil moisture
# ---------------------------
plt.figure(figsize=(12,5))
plt.plot(df['date'], df['predicted_sm'], marker='o', linestyle='-', label='Predicted SM')
plt.plot(df['date'], df['sm'], marker='x', linestyle='--', label='Actual SMAP', alpha=0.8)
plt.title('Dire Dawa: ML-Estimated Soil Moisture (3-day)')
plt.xlabel('Date')
plt.ylabel('Soil Moisture (m³/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
