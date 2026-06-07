import requests
import tqdm
import os

def download_files(local_path, start_year, end_year, path, prefix):
    base_url = f"https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/{path}/"

    for year in tqdm.tqdm(range(start_year, end_year + 1), desc=f"Downloading {path} files"):
        file_name = f"{prefix}_hyras_1_{year}_v6-1_de.nc"
        if os.path.exists(f"{local_path}/{file_name}"):
            print(f"File already exists: {file_name}, skipping download.")
            continue
        file_url = f"{base_url}/{file_name}"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(f"{local_path}/{file_name}", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: {file_name}, Status code: {response.status_code}")

def download_average_temperature_files(local_path):
    download_files(local_path, 1951, 2023, "air_temperature_mean", "tas")

def download_precipitation_files(local_path):
    download_files(local_path, 1951, 2023, "precipitation", "pr")

def download_min_average_temperature_files(local_path):
    download_files(local_path, 1951, 2023, "air_temperature_minimum", "tasmin")

def download_max_average_temperature_files(local_path):
    download_files(local_path, 1951, 2023, "air_temperature_maximum", "tasmax")

if __name__ == "__main__":
    download_average_temperature_files("../.data/")
    download_precipitation_files("../.data/")
    download_min_average_temperature_files("../.data/")
    download_max_average_temperature_files("../.data/")