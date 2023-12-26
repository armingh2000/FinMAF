import pandas as pd
import src.configs as configs
from pyspark.sql.functions import sin, cos, month, dayofmonth, year
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
import os
from tqdm import tqdm


def get_stock_metadata(logger):
    logger.info("Getting valid stock metadata ...")

    metadata = pd.read_csv(
        configs.valid_stocks_metadata,
        dtype={"Symbol": str, "NASDAQ Symbol": str},
        keep_default_na=False,  # This prevents Pandas from interpreting "NA" as NaN
    )

    return metadata


def normalize_and_create_features(df):
    first_element = udf(lambda v: float(v[0]), DoubleType())

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
        scaler = MinMaxScaler(inputCol=col + "_Vec", outputCol=col + "_Scaled_Value")
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        # Drop the original and vector columns, keep the scaled one
        # df = df.drop(col).drop(col + "_Vec").withColumnRenamed(col + "_Scaled", col)
        df = df.withColumn(col + "_Scaled", first_element(col + "_Scaled_Value"))
        df = df.drop(col + "_Vec").drop(col + "_Scaled_Value")

    return df


def dump_normalized_dataset(metadata, spark):
    for symbol in tqdm(metadata["Symbol"]):
        file_path = os.path.join(configs.dps_clean, f"{symbol}.csv")
        df = spark.read.csv(file_path, header=True, schema=configs.data_schema)

        # Normalize and create cyclic features
        df = normalize_and_create_features(df)

        # Dump normalized data to CSV
        normalized_file_path = os.path.join(configs.mt_normalized, f"{symbol}.csv")
        df.write.mode("overwrite").csv(normalized_file_path, header=True)
