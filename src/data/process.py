import pandas as pd
import os
import src.configs as configs
from tqdm import tqdm
import src.utils as utils
import math


def load_raw_symbols():
    symbols = []
    n_skip = 0

    for file in tqdm(os.listdir(configs.data_raw)):
        if file.endswith(".csv"):
            df = pd.read_csv(
                configs.data_raw / file,
                dtype={
                    "Open": "float",
                    "High": "float",
                    "Low": "float",
                    "Close": "float",
                    "Volume": "int",
                },
                parse_dates=["Date"],
                index_col="Date",
            )

            df.drop("Repaired?", axis=1, inplace=True)

            if len(df) > configs.window_size:
                symbol = utils.extract_filename_without_extension(file)
                symbols.append([df, symbol])

            else:
                print(f"Skipped {file} with length {len(df)}")
                n_skip += 1

    print(f"Skipped {n_skip} symbols...")
    print(f"Loaded {len(symbols)} symbols for processing...")

    return symbols


def process_symbols(symbols):
    processed_symbols = []

    for df, symbol in tqdm(symbols):
        rbf, _, _ = utils.get_scalers(df)

        transformed_features = rbf.transform(df)
        for i in range(transformed_features.shape[1]):
            df[f"rbf_{i}"] = transformed_features[:, i]

        processed_symbols.append(df)
        df.to_csv(configs.data_processed / f"{symbol}.csv")

    print(f"Processed {len(symbols)} symbols...")

    return symbols
