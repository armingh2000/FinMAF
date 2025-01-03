import yfinance as yf
import pandas as pd
import src.configs as configs
from tqdm import tqdm
import sys

# silent SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def download_symbols():
    data = pd.read_csv(
        configs.metadata_url,
        sep="|",
        dtype={"Symbol": str, "Symbol": str},
        keep_default_na=False,  # This prevents Pandas from interpreting "NA" as NaN
    )
    data_clean = data[data["Test Issue"] == "N"]

    # remove symbols with signs
    data_clean.drop(
        data_clean[data_clean["Symbol"].apply(lambda x: not x.isalpha())].index,
        inplace=True,
    )

    symbols = data_clean["Symbol"].tolist()
    print("total number of symbols traded after filtering = {}".format(len(symbols)))

    return symbols, data_clean


def save_symbols(symbols, data_clean):
    limit = configs.limit if configs.limit else len(symbols)
    end = min(configs.offset + limit, len(symbols))
    is_valid = [False] * len(symbols)
    is_failed = [False] * len(symbols)

    for i in tqdm(range(configs.offset, end)):
        s = symbols[i]

        attempts = 0
        while attempts < 3:
            try:
                df = yf.download(
                    s,
                    repair=configs.yf_repair,
                    end=configs.end_date,
                    progress=configs.yf_progress_bar,
                    rounding=configs.yf_rounding,
                )

            except KeyError:
                # Increment the attempt counter if a KeyError occurs
                print(f"KeyError occurred for {s}")
                attempts += 1
                if attempts == 3:
                    break

            else:
                break

        if len(df.index) == 0:
            is_failed[i] = True
            continue

        # Check for corrupted data (non-integer values)
        df.columns = df.columns.droplevel(1)
        df.columns = ["Close", "High", "Low", "Open", "Repaired?", "Volume"]
        corrupted_indices = df["Volume"].apply(lambda x: x != int(x))

        # Strategy 1: Replace corrupted data with a default value, e.g., 0
        # df.loc[corrupted_indices, 'Volume'] = 0

        # Strategy 2: Remove rows with corrupted data
        # df = df[~corrupted_indices]

        # Strategy 3: Round to nearest integer
        df.loc[corrupted_indices, "Volume"] = df.loc[
            corrupted_indices, "Volume"
        ].round()
        df["Volume"] = df["Volume"].astype("int64")

        is_valid[i] = True
        df.to_csv(configs.data_raw / f"{s}.csv")

    valid_data = data_clean[is_valid]
    valid_data.to_csv(configs.valid_stocks_metadata, index=False)
    failed_data = data_clean[is_failed]
    failed_data.to_csv(configs.failed_stocks_metadata, index=False)

    print("Total number of failed symbols = {}".format(sum(is_failed)))
    print("Total number of valid symbols downloaded = {}".format(sum(is_valid)))

    return valid_data
