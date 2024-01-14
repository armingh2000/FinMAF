from torch.utils.data import Dataset
import src.configs as configs
from data import dump_stock_durations
from src.utils import load_dictionary
import os
import torch
from pyspark.sql.functions import col
import numpy as np
from torch.utils.data import random_split
from src.model.train.utils import dump_torch_object, load_torch_object


class StockHistoryDataset(Dataset):
    def __init__(self, metadata, spark, logger):
        self.metadata = metadata
        self.spark = spark
        self.logger = logger
        self.lstm_sequence_length = configs.lstm_sequence_length
        self.durations = self.load_durations()
        self.stock_history_dir = configs.mt_normalized

        if not self.durations:
            self.logger.info("No duration data found")
            self.durations = dump_stock_durations(metadata, spark, logger)

        self.data = self.calculate_data()
        self.set_input_columns()

    def set_input_columns(self):
        time_cols = ["Year_Scaled", "Month_sin", "Month_cos", "Day_sin", "Day_cos"]
        stock_cols = [
            "Open_Scaled",
            "High_Scaled",
            "Low_Scaled",
            "Close_Scaled",
            "Adj Close_Scaled",
            "Volume_Scaled",
            # "Row_number",
        ]
        self.input_columns = time_cols + stock_cols

    def calculate_data(self):
        self.logger.info("Calculating dataset data ...")
        data = []

        for symbol, duration in self.durations.items():
            n_interval = np.ceil(duration / self.lstm_sequence_length)
            for i in range(1, int(n_interval)):
                data.append((symbol, i * self.lstm_sequence_length + 1))

        return data

    def load_durations(self):
        try:
            duration = load_dictionary(
                configs.stock_durations_path, self.logger, "durations"
            )

        except ValueError:
            duration = None

        return duration

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        symbol, entry = self.data[idx]

        # Load the stock history data from a CSV file
        file_path = os.path.join(self.stock_history_dir, f"{symbol}.csv")
        df = self.spark.read.csv(
            file_path,
            header=True,
            schema=configs.data_schema,
        )

        # Select the relevant rows and columns
        start_idx = entry - self.lstm_sequence_length
        end_idx = entry - 1
        selected_data = df.filter(
            (col("Row_number") >= start_idx) & (col("Row_number") <= end_idx)
        ).select(self.input_columns)

        # Convert to PyTorch tensor
        sequence_data = [
            torch.tensor(row, dtype=torch.float32) for row in selected_data.collect()
        ]
        sequence_tensor = torch.stack(sequence_data)

        # Fetch current entry data
        current_entry = (
            df.filter(col("Row_number") == entry).select(self.input_columns).collect()
        )

        current_entry_tensor = torch.tensor(current_entry[0], dtype=torch.float32)

        return sequence_tensor, current_entry_tensor


def prepare_loaders(metadata, spark, logger):
    SHD = StockHistoryDataset(metadata, spark, logger)
    train_dataset, val_dataset, test_dataset = chunk_dataset(SHD)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=configs.shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=configs.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def chunk_dataset(dataset):
    train_size = int(configs.train_split * len(dataset))
    val_size = int(configs.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset
