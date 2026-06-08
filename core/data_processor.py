import pandas as pd
import os
from data_reader import monthly_total_germany, yearly_potato_yield
from data_reader import yearly_raps_yield
import tqdm

def create_big_data_file() -> pd.DataFrame:
    df = pd.DataFrame(columns=["year", "month", "monthly_total_precipitation", "rapeseed_yield", "potato_yield"])

    for filename in tqdm.tqdm(os.listdir("../.data/")):
        if filename.endswith(".nc") and "pr" in filename:
            year = filename.split("_")[3]
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            monthly_total_precipitation = monthly_total_germany(os.path.join("../.data/", filename))
            try:
                rape_yield = yearly_raps_yield("../.data/41241-0002_de_flat.csv", year)
            except ValueError as e:
                print(f"Error occurred while reading rapeseed yield for year {year}: {e}")
                rape_yield = np.nan

            try:
                potato_yield = yearly_potato_yield("../.data/41241-0002_de_flat.csv", year)
            except ValueError as e:
                print(f"Error occurred while reading potato yield for year {year}: {e}")
                potato_yield = np.nan

            #print(len(months), len(monthly_total_precipitation[1]))
            df.loc[len(df)] = [year, months, monthly_total_precipitation[1].tolist(), rape_yield, potato_yield]

        df.to_csv("../.data/big_hyras_file.csv", index=False)
    return df


if __name__ == "__main__":
    create_big_data_file()