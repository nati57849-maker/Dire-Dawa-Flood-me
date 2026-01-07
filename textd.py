import requests
import os

output_folder = r"C:\Users\gfeka\Desktop\soil_rasters"
os.makedirs(output_folder, exist_ok=True)

rasters = {
    "clay_0-5cm.tif": "https://files.isric.org/soilgrids/latest/clay/clay_0-5cm.tif",
    "silt_0-5cm.tif": "https://files.isric.org/soilgrids/latest/silt/silt_0-5cm.tif",
    "sand_0-5cm.tif": "https://files.isric.org/soilgrids/latest/sand/sand_0-5cm.tif"
}

for filename, url in rasters.items():
    file_path = os.path.join(output_folder, filename)
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"{filename} downloaded successfully at {file_path}!")
