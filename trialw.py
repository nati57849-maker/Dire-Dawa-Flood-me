from datetime import datetime
import pandas as pd
from meteostat import Point, Daily

# Define location: Dire Dawa, Ethiopia
dire_dawa = Point(9.6000, 41.8661, 1260)  # lat, lon, elevation

# 30-year baseline period
start = datetime(1991, 1, 1)
end = datetime(2020, 12, 31)

# Fetch daily data
data = Daily(dire_dawa, start, end).fetch()

# Group by month → compute averages
baseline = data.groupby(data.index.month).agg({
    'tavg': 'mean',
    'prcp': 'mean'
}).rename_axis('month')

# Convert months to names
baseline.index = pd.to_datetime(baseline.index, format='%m').month_name()

print(baseline.round(2))
