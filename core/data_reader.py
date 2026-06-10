import numpy as np
import pandas as pd
import os
import netCDF4

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '.data')


# --- NetCDF helpers ---

def print_metadata(filename: str) -> None:
    with netCDF4.Dataset(filename, 'r') as ds:
        print("Metadata:")
        for var_name in ds.variables:
            var = ds.variables[var_name]
            print(f"Variable: {var_name}")
            print(f"  Dimensions: {var.dimensions}")
            print(f"  Shape: {var.shape}")
            print(f"  Data type: {var.dtype}")
            if var_name == 'pr':
                print(f"  Units: {var.units}")

def read_precipitation(filename: str) -> np.ndarray:
    with netCDF4.Dataset(filename, 'r') as ds:
        return ds.variables['pr'][:]

def read_mean_temperature(filename: str) -> np.ndarray:
    with netCDF4.Dataset(filename, 'r') as ds:
        return ds.variables['tas'][:]

def read_time(filename: str) -> np.ndarray:
    with netCDF4.Dataset(filename, 'r') as ds:
        time_var = ds.variables['time']
        return np.array(netCDF4.num2date(time_var[:], time_var.units))

def _months_index(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    months = np.array([t.strftime('%Y-%m') for t in times])
    return np.unique(months, return_inverse=True)


# --- Precipitation ---

def total_precipitation(filename: str) -> float:
    return float(np.sum(read_precipitation(filename)))

def average_precipitation_germany(filename: str) -> float:
    return float(np.mean(read_precipitation(filename)))

def daily_average_precipitation_germany(filename: str) -> np.ndarray:
    return np.mean(read_precipitation(filename), axis=(1, 2))

def monthly_average_precipitation_germany(filename: str) -> np.ndarray:
    daily_mean = daily_average_precipitation_germany(filename)
    unique_months, inverse = _months_index(read_time(filename))
    monthly_mean = np.array([daily_mean[inverse == i].mean() for i in range(len(unique_months))])
    return monthly_mean

def monthly_total_precipitation_germany(filename: str) -> np.ndarray:
    daily_mean = daily_average_precipitation_germany(filename)
    unique_months, inverse = _months_index(read_time(filename))
    monthly_total = np.array([daily_mean[inverse == i].sum() for i in range(len(unique_months))])
    return monthly_total

def get_certain_months_total_precipitation_germany(filename: str, months: list[int]) -> np.ndarray:
    out = []
    full_year_monthly_totals = monthly_total_precipitation_germany(filename=filename)
    for month_index in months:
        out.append(full_year_monthly_totals[month_index-1])

    return out


# --- Temperature ---

def average_temperature(filename: str) -> float:
    return float(np.mean(read_mean_temperature(filename)))

def average_monthly_temperature(filename: str) -> np.ndarray:
    daily_mean = np.mean(read_mean_temperature(filename), axis=(1, 2))
    unique_months, inverse = _months_index(read_time(filename))
    monthly_mean = np.array([daily_mean[inverse == i].mean() for i in range(len(unique_months))])
    return monthly_mean


# --- Crop yield ---

def _load_yield_csv_combined() -> pd.DataFrame:
    df1 = pd.read_csv(os.path.join(DATA_DIR, "41241-0002_de_flat.csv"), delimiter=";")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";")
    return pd.concat([df1, df2], ignore_index=True)

def _load_yield_per_hectare_csv() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_DIR, "41241-0010_de_flat.csv"),
        delimiter=";", decimal=",", na_values=["-", "/", "."],
    )

def _yearly_yield(crop_label: str, year: int, crop_name: str) -> float:
    df = _load_yield_csv_combined()
    rows = df[
        (df["2_variable_attribute_label"] == crop_label) &
        (df["time"].astype(str) == str(year))
    ]
    if rows.empty:
        raise ValueError(f"No {crop_name} yield data found for year {year}")
    value = rows["value"].iloc[0]
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"{crop_name} yield value for year {year} is not numeric: {value}")

def _yearly_yield_per_hectare(crop_pattern: str, year: int, crop_name: str) -> float:
    df = _load_yield_per_hectare_csv()
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains(crop_pattern, case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    rows = df[mask]
    if rows.empty:
        raise ValueError(f"No {crop_name} yield data found for year {year}")
    numeric = pd.to_numeric(rows["value"], errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"{crop_name} yield values for year {year} are missing or non-numeric")
    return float(numeric.mean())

def yearly_potato_yield(year: int) -> float:
    return _yearly_yield("Kartoffeln", year, "potato")

def yearly_raps_yield(year: int) -> float:
    return _yearly_yield("Winterraps", year, "rapeseed")

def yearly_raps_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Winterraps", year, "rapeseed")

def yearly_soy_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Sojabohnen", year, "soybean")

def yearly_oats_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Hafer", year, "oat")

def yearly_wheat_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Weizen", year, "wheat")

def yearly_barley_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Gerste", year, "barley")

def yearly_rye_yield_per_hectare(year: int) -> float:
    return _yearly_yield_per_hectare("Roggen und Wintermenggetreide", year, "rye")


if __name__ == "__main__":
    #print_metadata(os.path.join(DATA_DIR, "tas_hyras_1_2023_v6-1_de.nc"))
    #print("Average temperature:", average_temperature(os.path.join(DATA_DIR, "tas_hyras_1_2023_v6-1_de.nc")))
    #print("Average monthly temperature:", average_monthly_temperature(os.path.join(DATA_DIR, "tas_hyras_1_2023_v6-1_de.nc")))
    print(get_certain_months_total_precipitation_germany("../.data/pr_hyras_1_1952_v6-1_de.nc", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
