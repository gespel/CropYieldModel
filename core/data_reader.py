import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import netCDF4
import tqdm
import pathlib

# data directory relative to this module
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '.data')

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
            pr = ds.variables['pr'][:]
        return pr

def read_mean_temperature(filename: str) -> np.ndarray:
        with netCDF4.Dataset(filename, 'r') as ds:
            tas = ds.variables['tas'][:]
        return tas

def total_precipitation(filename: str) -> float:
    pr = read_precipitation(filename)
    return float(np.sum(pr))

def average_temperature(filename: str) -> float:
    tas = read_mean_temperature(filename)
    return float(np.mean(tas))

def average_precipitation_germany(filename: str) -> float:
    pr = read_precipitation(filename)
    return float(np.mean(pr))

def daily_average_precipitation_germany(filename: str) -> np.ndarray:
    pr = read_precipitation(filename)
    return np.mean(pr, axis=(1, 2))

def read_time(filename: str) -> np.ndarray:
    with netCDF4.Dataset(filename, 'r') as ds:
        time_var = ds.variables['time']
        return np.array(netCDF4.num2date(time_var[:], time_var.units))

def monthly_average_precipitation_germany(filename: str) -> tuple[list[str], np.ndarray]:
    daily_mean = daily_average_precipitation_germany(filename)
    times = read_time(filename)
    months = np.array([t.strftime('%Y-%m') for t in times])
    unique_months, inverse = np.unique(months, return_inverse=True)
    monthly_mean = np.array([daily_mean[inverse == i].mean() for i in range(len(unique_months))])
    return unique_months.tolist(), monthly_mean

def monthly_total_precipitation_germany(filename: str) -> tuple[list[str], np.ndarray]:
    daily_mean = daily_average_precipitation_germany(filename)
    times = read_time(filename)
    months = np.array([t.strftime('%Y-%m') for t in times])
    unique_months, inverse = np.unique(months, return_inverse=True)
    monthly_total = np.array([daily_mean[inverse == i].sum() for i in range(len(unique_months))])
    return unique_months.tolist(), monthly_total

def yearly_potato_yield(year: int) -> float:
    df1 = pd.read_csv(os.path.join(DATA_DIR, "41241-0002_de_flat.csv"), delimiter=";")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";")
    df = pd.concat([df1, df2], ignore_index=True)

    potato_rows = df[df["2_variable_attribute_label"] == "Kartoffeln"]
    year_rows = potato_rows[potato_rows["time"].astype(str) == str(year)]

    if year_rows.empty:
        raise ValueError(f"No potato yield data found for year {year}")

    return float(year_rows["value"].iloc[0].replace(",", "."))

def yearly_raps_yield(year: int) -> float:
    df1 = pd.read_csv(os.path.join(DATA_DIR, "41241-0002_de_flat.csv"), delimiter=";")
    df2 = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";")
    df = pd.concat([df1, df2], ignore_index=True)

    raps_rows = df[df["2_variable_attribute_label"] == "Winterraps"]
    year_rows = raps_rows[raps_rows["time"].astype(str) == str(year)]

    if year_rows.empty:
        raise ValueError(f"No rapeseed yield data found for year {year}")
    
    try:
        return float(year_rows["value"].iloc[0].replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"Yield value for year {year} is not numeric: {year_rows['value'].iloc[0]}")
    

def yearly_raps_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Raps in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Winterraps", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    raps_rows = df[mask].copy()

    if raps_rows.empty:
        raise ValueError(f"No rapeseed yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    raps_rows["value_num"] = pd.to_numeric(raps_rows["value"], errors="coerce")
    raps_rows = raps_rows.dropna(subset=["value_num"])

    if raps_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(raps_rows["value_num"].mean())

def yearly_soy_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Soja in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Sojabohnen", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    soy_rows = df[mask].copy()

    if soy_rows.empty:
        raise ValueError(f"No soybean yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    soy_rows["value_num"] = pd.to_numeric(soy_rows["value"], errors="coerce")
    soy_rows = soy_rows.dropna(subset=["value_num"])

    if soy_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(soy_rows["value_num"].mean())

def yearly_oats_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Hafer in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Hafer", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    oats_rows = df[mask].copy()

    if oats_rows.empty:
        raise ValueError(f"No oat yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    oats_rows["value_num"] = pd.to_numeric(oats_rows["value"], errors="coerce")
    oats_rows = oats_rows.dropna(subset=["value_num"])

    if oats_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(oats_rows["value_num"].mean())

def yearly_wheat_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Weizen in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Weizen", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    wheat_rows = df[mask].copy()

    if wheat_rows.empty:
        raise ValueError(f"No wheat yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    wheat_rows["value_num"] = pd.to_numeric(wheat_rows["value"], errors="coerce")
    wheat_rows = wheat_rows.dropna(subset=["value_num"])

    if wheat_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(wheat_rows["value_num"].mean())

def yearly_barley_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Gerste in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Gerste", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    barley_rows = df[mask].copy()

    if barley_rows.empty:
        raise ValueError(f"No barley yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    barley_rows["value_num"] = pd.to_numeric(barley_rows["value"], errors="coerce")
    barley_rows = barley_rows.dropna(subset=["value_num"])

    if barley_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(barley_rows["value_num"].mean())

def yearly_rye_yield_per_hectare(year: int) -> float:
    # Read CSV with proper decimal parsing and common NA tokens
    df = pd.read_csv(os.path.join(DATA_DIR, "41241-0010_de_flat.csv"), delimiter=";", decimal=",", na_values=["-", "/", "."]) 

    # Filter: rows that mention Roggen in the 2_variable_attribute_label,
    # that are 'Ertrag je Hektar' and belong to the requested year
    mask = (
        df["2_variable_attribute_label"].astype(str).str.contains("Roggen und Wintermenggetreide", case=False, na=False)
        & (df["value_variable_label"] == "Ertrag je Hektar")
        & (df["time"].astype(str) == str(year))
    )
    rye_rows = df[mask].copy()

    if rye_rows.empty:
        raise ValueError(f"No rye yield data found for year {year}")

    # Convert 'value' to numeric and drop non-numeric entries
    rye_rows["value_num"] = pd.to_numeric(rye_rows["value"], errors="coerce")
    rye_rows = rye_rows.dropna(subset=["value_num"])

    if rye_rows.empty:
        raise ValueError(f"Yield values for year {year} are missing or non-numeric")

    # Return average yield per hectare across regions for the year
    return float(rye_rows["value_num"].mean())


if __name__ == "__main__":
    print_metadata(os.path.join(DATA_DIR, "tas_hyras_1_2023_v6-1_de.nc"))