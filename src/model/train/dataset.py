import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import sin, cos, month, dayofmonth, year
from pyspark.sql.window import Window
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
import src.configs as configs
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from tqdm import tqdm


class StockHistoryDataset(Dataset):
    def __init__(self, metadata, pca_embeddings, sequence_length=60):
        self.metadata = metadata
        self.pca_embeddings = pca_embeddings
        self.stock_history_dir = configs.dps_clean
        self.sequence_length = sequence_length
        self.spark = SparkSession.builder.appName("StockHistoryDataset").getOrCreate()
        self.data = self.preprocess_data()

    def preprocess_data(self):
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
                Window.partitionBy()
                .orderBy("Date")
                .rowsBetween(-self.sequence_length, -1)
            )

            for col_name in row_cols:
                df = df.withColumn(
                    col_name + "_seq", collect_list(col_name).over(window)
                )

            # Convert to an RDD for generating sequences
            rdd = df.rdd.map(
                lambda row: (
                    [row[col] for col in seq_cols],
                    [row[col] for col in row_cols],
                )
            )

            # Collect sequences and targets
            sequences = rdd.collect()[1:]
            data.extend(sequences)

        return data

    def normalize_and_create_features(
        self,
        df,
    ):
        # Adding time features
        df = df.withColumn("Year", year("Date"))
        df = df.withColumn("Month_sin", sin(2 * np.pi * month("Date") / 12))
        df = df.withColumn("Month_cos", cos(2 * np.pi * month("Date") / 12))
        df = df.withColumn("Day_sin", sin(2 * np.pi * dayofmonth("Date") / 31))
        df = df.withColumn("Day_cos", cos(2 * np.pi * dayofmonth("Date") / 31))

        # Normalize numerical features
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Year"]:
            # Convert column to a vector
            assembler = VectorAssembler(inputCols=[col], outputCol=col + "_Vec")
            df = assembler.transform(df)

            # Apply MinMaxScaler
            scaler = MinMaxScaler(
                inputCol=col + "_Vec", outputCol=col + "_Scaled_Value"
            )
            scaler_model = scaler.fit(df)
            df = scaler_model.transform(df)

            # Drop the original and vector columns, keep the scaled one
            # df = df.drop(col).drop(col + "_Vec").withColumnRenamed(col + "_Scaled", col)
            df = df.withColumn(
                col + "_Scaled", self.first_element(col + "_Scaled_Value")
            )
            df = df.drop(col + "_Vec").drop(col + "_Scaled_Value")

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        # sequence.shape = (n, 11, sequence_length)
        symbol = sequence[0, 0]  # Assuming the first column is 'Symbol'
        sequence_embedding = self.pca_embeddings[symbol]
        return (
            torch.tensor(sequence_embedding, dtype=torch.float),
            torch.tensor(sequence[:, 1:], dtype=torch.float),  # Exclude 'Symbol' column
            torch.tensor(target[1:], dtype=torch.float),  # Exclude 'Symbol' column
        )


# Usage
# metadata = ...  # Load your metadata DataFrame
# pca_embeddings = ...  # Load your PCA embeddings
# dataset = StockHistoryDataset(metadata, pca_embeddings)
