from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
    LongType,
    BooleanType,
)
import pandas as pd
from tqdm import tqdm
import os
import src.configs as configs


def get_stock_metadata():
    metadata = pd.read_csv(configs.stocks_metadata)

    return metadata


def get_embedding_input(metadata):
    inputs = []

    for _, row in tqdm(metadata.iterrows()):
        inputs.append([row["NASDAQ Symbol"], row["Security Name"]])

    return inputs
