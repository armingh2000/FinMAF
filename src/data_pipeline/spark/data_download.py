import pandas as pd
import yfinance as yf
import src.configs as configs
from src.utils import mkpath
from tqdm import tqdm

# silent SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def download_symbols(logger):
    data = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep="|"
    )
    data_clean = data[data["Test Issue"] == "N"]
    data_clean["NASDAQ Symbol"] = data_clean["NASDAQ Symbol"].astype(str)
    # remove symbols with signs
    data_clean.drop(
        data_clean[data_clean["NASDAQ Symbol"].apply(lambda x: not x.isalpha())].index,
        inplace=True,
    )

    symbols = data_clean["NASDAQ Symbol"].tolist()
    logger.info(
        "total number of symbols traded after filtering = {}".format(len(symbols))
    )

    return symbols, data_clean


def save_symbols(symbols, data_clean, logger):
    limit = configs.limit if configs.limit else len(symbols)
    end = min(configs.offset + limit, len(symbols))
    is_valid = [False] * len(symbols)
    is_failed = [False] * len(symbols)

    for i in tqdm(range(configs.offset, end)):
        s = symbols[i]

        df = yf.download(
            s,
            repair=True,
            end=configs.end_date,
            progress=configs.yfinance_progress_bar,
            rounding=True,
        )

        if len(df.index) == 0:
            is_failed[i] = True
            continue

        # Check for corrupted data (non-integer values)
        corrupted_indices = df["Volume"].apply(lambda x: x != int(x))

        # Strategy 1: Replace corrupted data with a default value, e.g., 0
        # df.loc[corrupted_indices, 'Volume'] = 0

        # Strategy 2: Remove rows with corrupted data
        # df = df[~corrupted_indices]

        # Strategy 3: Round to nearest integer
        df.loc[corrupted_indices, "Volume"] = df.loc[
            corrupted_indices, "Volume"
        ].round()

        is_valid[i] = True
        df.to_csv(configs.dps_raw / f"{s}.csv")

    valid_data = data_clean[is_valid]
    valid_data.to_csv(configs.meta_file_path / "symbols_valid_meta.csv", index=False)
    failed_data = data_clean[is_failed]
    failed_data.to_csv(configs.meta_file_path / "symbols_failed_meta.csv", index=False)

    logger.info("Total number of failed symbols = {}".format(sum(is_failed)))
    logger.info("Total number of valid symbols downloaded = {}".format(sum(is_valid)))

    return valid_data
