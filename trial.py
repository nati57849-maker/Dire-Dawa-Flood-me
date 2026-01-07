import requests
import pandas as pd
from datetime import datetime
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
LAT = 9.6000          # Dire Dawa latitude
LON = 41.8661         # Dire Dawa longitude
API_KEY = "26b7d5e90ade2e48ee657a85d31d5730"  # Replace with your OpenWeatherMap API key
EXCLUDE = "minutely,alerts,current"
UNITS = "metric"
RAIN_THRESHOLD = 20  # mm/hour for flood alert
FETCH_INTERVAL = 3600  # seconds (1 hour)

# Function to fetch data
def fetch_weather():
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat=9.6000&lon=41.8661&exclude=minutely,alerts,current&appid=YOUR_API_KEY&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# Function to process hourly data and check flood risk
def process_hourly(data):
    hourly_data = []
    alert_triggered = False

    for hour in data['hourly']:
        dt = datetime.fromtimestamp(hour['dt'])
        temp = hour['temp']
        rain = hour.get('rain', {}).get('1h', 0)
        wind = hour['wind_speed']
        hourly_data.append({
            'datetime': dt,
            'temp_C': temp,
            'rain_mm': rain,
            'wind_m_s': wind
        })
        
        # Flood alert
        if rain >= RAIN_THRESHOLD:
            alert_triggered = True
            print(f"⚠️ FLOOD ALERT! Heavy rain forecasted at {dt}: {rain} mm/hour")

    df_hourly = pd.DataFrame(hourly_data)
    df_hourly.to_csv('dire_dawa_hourly_weather.csv', index=False)
    print("Hourly data saved to 'dire_dawa_hourly_weather.csv'.")
    
    if not alert_triggered:
        print("No flood risk detected in the upcoming hours.")
    
    return df_hourly

# Function to process daily data
def process_daily(data):
    daily_data = []
    for day in data['daily']:
        dt = datetime.fromtimestamp(day['dt'])
        temp_day = day['temp']['day']
        rain = day.get('rain', 0)
        wind = day['wind_speed']
        daily_data.append({
            'date': dt.date(),
            'temp_day_C': temp_day,
            'rain_mm': rain,
            'wind_m_s': wind
        })
    df_daily = pd.DataFrame(daily_data)
    df_daily.to_csv('dire_dawa_daily_weather.csv', index=False)
    print("Daily data saved to 'dire_dawa_daily_weather.csv'.")
    return df_daily

# ----------------------------
# MAIN LOOP: Run every hour
# ----------------------------
while True:
    print(f"\nFetching weather data at {datetime.now()}...")
    weather_data = fetch_weather()
    if weather_data:
        process_hourly(weather_data)
        process_daily(weather_data)
    print(f"Waiting {FETCH_INTERVAL/60} minutes for the next fetch...")
    time.sleep(FETCH_INTERVAL)
