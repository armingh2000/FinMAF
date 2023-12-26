import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import src.configs as configs
from pyspark.sql.functions import collect_list
from tqdm import tqdm
from utils import dump_dataset_data, load_dataset_data
from src.utils import mkpath


class StockHistoryDataset(Dataset):
    def __init__(self, metadata, pca_embeddings, logger):
        self.metadata = metadata
        self.pca_embeddings = pca_embeddings
        self.logger = logger
        self.stock_history_dir = configs.dps_clean
        self.lstm_sequence_length = configs.lstm_sequence_length

        self.logger.info("Creating Spark session ...")
        mkpath(configs.mt_spark_log_path)
        self.spark = (
            SparkSession.builder.appName("StockHistoryDataset")
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", configs.mt_spark_log_path)
            .config("spark.executor.extraJavaOptions", "-XX:+UseParallelGC")
            .config("spark.driver.extraJavaOptions", "-XX:+UseParallelGC")
            .getOrCreate()
        )

        self.init_udfs()

        self.data = self.load_data()
        if not self.data:
            self.data = self.preprocess_data()
            self.dump_data()

    def preprocess_data(self):
        self.logger.info("Preprocessing dataset data ...")

        data = []
        time_cols = ["Year_Scaled", "Month_sin", "Month_cos", "Day_sin", "Day_cos"]
        stock_cols = [
            "Open_Scaled",
            "High_Scaled",
            "Low_Scaled",
            "Close_Scaled",
            "Adj Close_Scaled",
            "Volume_Scaled",
        ]
        row_cols = time_cols + stock_cols
        seq_cols = [f + "_seq" for f in row_cols]

        for symbol in tqdm(self.metadata["Symbol"]):
            file_path = os.path.join(self.stock_history_dir, f"{symbol}.csv")
            df = self.spark.read.csv(file_path, header=True, schema=configs.data_schema)

            # Normalize and create cyclic features
            df = self.normalize_and_create_features(df)

            # Create sequences using a sliding window approach
            window = (
                Window.partitionBy("Year_Scaled")
                .orderBy("Date")
                .rowsBetween(-self.lstm_sequence_length, -1)
            )

            for col_name in row_cols:
                df = df.withColumn(
                    col_name + "_seq", collect_list(col_name).over(window)
                )

            # Convert to an RDD for generating sequences
            rdd = df.rdd.map(
                lambda row: (
                    np.array([row[col] for col in seq_cols]),
                    np.array([row[col] for col in row_cols]),
                )
            )

            # Collect sequences and targets
            sequences = rdd.collect()[1:]
            data.extend(sequences)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[
            idx
        ]  # sequence.shape = (11, lstm_sequence_length), target.shape = (11, 1)

        symbol = self.metadata.iloc[idx]["Symbol"]
        sequence_embedding = self.pca_embeddings[symbol]

        return (
            torch.tensor(sequence_embedding, dtype=torch.float),
            torch.tensor(sequence, dtype=torch.float),
            torch.tensor(target, dtype=torch.float),
        )

    def dump_data(self):
        self.logger.info("Dumping dataset data ...")
        dump_dataset_data(self.data)

    def load_data(self):
        self.logger.info("Loading dataset data ...")

        if not os.path.exists(configs.embedding_dataset_data_path):
            self.logger.info(
                f"Filepath {configs.embedding_dataset_data_path} does not exist. Making a new dataset ..."
            )
            return None

        self.logger.info(
            f"Loading dataset data from {configs.embedding_dataset_data_path}"
        )
        return load_dataset_data()
