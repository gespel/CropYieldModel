import numpy as np
from matplotlib import pyplot as plt
import netCDF4

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

def average_precipitation_germany(filename: str) -> float:
    pr = read_precipitation(filename)
    return float(np.mean(pr))

def daily_average_germany(filename: str) -> np.ndarray:
    pr = read_precipitation(filename)
    return np.mean(pr, axis=(1, 2))

def read_time(filename: str) -> np.ndarray:
    with netCDF4.Dataset(filename, 'r') as ds:
        time_var = ds.variables['time']
        return np.array(netCDF4.num2date(time_var[:], time_var.units))

def monthly_average_germany(filename: str) -> tuple[list[str], np.ndarray]:
    daily_mean = daily_average_germany(filename)
    times = read_time(filename)
    months = np.array([t.strftime('%Y-%m') for t in times])
    unique_months, inverse = np.unique(months, return_inverse=True)
    monthly_mean = np.array([daily_mean[inverse == i].mean() for i in range(len(unique_months))])
    return unique_months.tolist(), monthly_mean

def monthly_total_germany(filename: str) -> tuple[list[str], np.ndarray]:
    daily_mean = daily_average_germany(filename)
    times = read_time(filename)
    months = np.array([t.strftime('%Y-%m') for t in times])
    unique_months, inverse = np.unique(months, return_inverse=True)
    monthly_total = np.array([daily_mean[inverse == i].sum() for i in range(len(unique_months))])
    return unique_months.tolist(), monthly_total
