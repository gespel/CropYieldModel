import requests
import tqdm

def download_average_temperature_files(local_path):
    base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/"

    for year in tqdm.tqdm(range(1951, 2024), desc="Downloading temperature files"):
        file_url = f"{base_url}/tas_hyras_1_{year}_v6-1_de.nc"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(f"{local_path}/tas_hyras_1_{year}_v6-1_de.nc", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: tas_hyras_1_{year}_v6-1_de.nc, Status code: {response.status_code}")

def download_precipitation_files(local_path):
    base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/"

    for year in tqdm.tqdm(range(1951, 2024), desc="Downloading precipitation files"):
        file_url = f"{base_url}/pr_hyras_1_{year}_v6-1_de.nc"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(f"{local_path}/pr_hyras_1_{year}_v6-1_de.nc", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: pr_hyras_1_{year}_v6-1_de.nc, Status code: {response.status_code}")

def download_min_average_temperature_files(local_path):
    base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_minimum/"

    for year in tqdm.tqdm(range(1951, 2024), desc="Downloading minimum temperature files"):
        file_url = f"{base_url}/tasmin_hyras_1_{year}_v6-1_de.nc"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(f"{local_path}/tasmin_hyras_1_{year}_v6-1_de.nc", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: tasmin_hyras_1_{year}_v6-1_de.nc, Status code: {response.status_code}")

def download_max_average_temperature_files(local_path):
    base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_maximum/"

    for year in tqdm.tqdm(range(1951, 2024), desc="Downloading maximum temperature files"):
        file_url = f"{base_url}/tasmax_hyras_1_{year}_v6-1_de.nc"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(f"{local_path}/tasmax_hyras_1_{year}_v6-1_de.nc", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: tasmax_hyras_1_{year}_v6-1_de.nc, Status code: {response.status_code}")

if __name__ == "__main__":
    download_average_temperature_files("../.data/")
    download_precipitation_files("../.data/")
    download_min_average_temperature_files("../.data/")
    download_max_average_temperature_files("../.data/")