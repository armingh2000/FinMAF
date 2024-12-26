import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import src.configs as configs
import pandas as pd
import src.utils as utils


class FinMAFDataset(Dataset):
    def __init__(self, dfs):
        self.dfs = dfs
        self.prepare_data()

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        pd.options.mode.chained_assignment = None  # default='warn'
        self.data = []

        print(f"Preparing dataset...")

        for df in tqdm(self.dfs):
            max_year = df.index.max().year

            # set year_window_df
            year_window_df = df[df.index.year >= max_year - 20]
            year_window_df = year_window_df.iloc[: -configs.window_size]

            grouped = year_window_df.groupby(
                [year_window_df.index.year, year_window_df.index.month]
            )
            random_days = grouped.sample(n=1, random_state=2)
            random_days.index = pd.DatetimeIndex(random_days.index)

            for index, row in random_days.iterrows():
                random_day_index = df.index.get_loc(index)
                end_index = random_day_index + configs.window_size

                input_and_target = df.iloc[random_day_index : end_index + 1]
                input_and_target = utils.scale_df(
                    input_and_target, for_prediction=False
                )

                window = input_and_target.iloc[:-1][
                    [
                        "Close_Scaled",
                        "High_Scaled",
                        "Low_Scaled",
                        "Open_Scaled",
                        "Volume_Scaled",
                        "rbf_0",
                        "rbf_1",
                        "rbf_2",
                        "rbf_3",
                        "rbf_4",
                        "rbf_5",
                        "rbf_6",
                        "rbf_7",
                        "rbf_8",
                        "rbf_9",
                        "rbf_10",
                        "rbf_11",
                    ]
                ]
                target = input_and_target.iloc[-1][
                    [
                        "Close_Scaled",
                        "High_Scaled",
                        "Low_Scaled",
                        "Open_Scaled",
                        "Volume_Scaled",
                    ]
                ]

                sample = torch.FloatTensor(window.to_numpy())
                sample_target = torch.FloatTensor(target.to_numpy())

                self.data.append([sample, sample_target])

    def __getitem__(self, idx):
        return self.data[idx]


def load_processed_symbols():
    symbols = []

    for file in tqdm(os.listdir(configs.data_processed)):
        if file.endswith(".csv"):
            df = pd.read_csv(
                configs.data_processed / file,
                dtype={
                    "Open": "float",
                    "High": "float",
                    "Low": "float",
                    "Close": "float",
                    "Volume": "int",
                    "rbf_0": "float",
                    "rbf_1": "float",
                    "rbf_2": "float",
                    "rbf_3": "float",
                    "rbf_4": "float",
                    "rbf_5": "float",
                    "rbf_6": "float",
                    "rbf_7": "float",
                    "rbf_8": "float",
                    "rbf_9": "float",
                    "rbf_10": "float",
                    "rbf_11": "float",
                },
                parse_dates=["Date"],
                index_col="Date",
            )

            symbols.append(df)

    print(f"Loaded {len(symbols)} symbols for Dataset...")

    return symbols
