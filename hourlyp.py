import requests
import pandas as pd
from datetime import datetime
import os
import time
import logging
import sys

# -------------------------------
# CONFIGURATION
# -------------------------------
CITY = "Dire Dawa"   # Human-readable location
LAT = 9.6000
LON = 41.8661
API_KEY = "cd8abc61a64e458abcf121843252908"   # Replace with your WeatherAPI key
INTERVAL = 3600  # seconds (3600 = 1 hour). Change if needed.
CSV_FILE = "dire_dawa_weather.csv"  # Single CSV file

# -------------------------------
# LOGGING SETUP
# -------------------------------
logging.basicConfig(
    filename="weather_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# FUNCTION TO FETCH DATA
# -------------------------------
def fetch_weather():
    """Fetch current weather data for Dire Dawa from WeatherAPI.com"""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LAT},{LON}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant fields
        weather = {
            "city": CITY,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_c": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "precip_mm": data["current"]["precip_mm"],   # rainfall in mm
            "wind_kph": data["current"]["wind_kph"],     # wind speed
            "condition": data["current"]["condition"]["text"],
            "cloud": data["current"]["cloud"]            # cloud coverage %
        }
        return weather
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

# -------------------------------
# FUNCTION TO SAVE DATA
# -------------------------------
def save_to_csv(weather):
    """Append weather data to a single CSV file."""
    df = pd.DataFrame([weather])
    if not os.path.isfile(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)  # create file with headers
    else:
        df.to_csv(CSV_FILE, mode="a", index=False, header=False)

# -------------------------------
# COUNTDOWN FUNCTION
# -------------------------------
def countdown(seconds):
    """Show a live countdown until the next fetch."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timeformat = f"{mins:02d}:{secs:02d}"
        sys.stdout.write(f"\rNext fetch in {timeformat}...")
        sys.stdout.flush()
        time.sleep(1)
    print("\rFetching now!")

# -------------------------------
# AUTOMATED LOOP
# -------------------------------
def start_collection():
    print(f"Starting automated weather data collection for {CITY}...")
    print(f"All data will be saved in: {CSV_FILE}")
    print(f"Fetch interval: {INTERVAL/60} minutes\n")
    
    while True:
        weather = fetch_weather()
        if weather:
            print("\nCollected:", weather)
            save_to_csv(weather)
            logging.info(f"Data collected: {weather}")
        else:
            print("\nFetch failed. Retrying in 1 minute...")
            countdown(60)
            continue
        
        countdown(INTERVAL)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    try:
        start_collection()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting safely...")
        logging.info("Process stopped manually.")
