from pyspark.sql import SparkSession
import os
import src.configs as configs
from tqdm import tqdm
import pandas as pd
from pyspark.sql.functions import to_date, col
from pyspark.sql.types import DoubleType, LongType
from pyspark.sql.window import Window
from pyspark.sql.functions import last


def fill_null_values(df):
    """
    Fills null values in the DataFrame by carrying forward the last known value.

    :param df: Input Spark DataFrame
    :return: DataFrame with null values filled
    """

    window_spec = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

    for col_name in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df = df.withColumn(
            col_name, last(df[col_name], ignorenulls=True).over(window_spec)
        )

    return df


def dump_nulls(logger, df, file_name, nulls_df):
    logger.info("Dumping null records for further processing.")

    # Find rows and columns with null values
    for col in df.columns:
        null_rows = df.filter(df[col].isNull()).collect()
        for row in null_rows:
            # Append results to nulls_df
            temp_df = pd.DataFrame(
                [[file_name, col, row["Date"]]],
                columns=["file_name", "column_with_null", "row_index"],
            )

            nulls_df = pd.concat(
                [nulls_df, temp_df],
                ignore_index=True,
            )
    return nulls_df


def clean_stock_data(spark, logger):
    # Initialize an empty DataFrame to store results
    nulls_df = pd.DataFrame(columns=["file_name", "column_with_null", "row_index"])

    # Iterate over each file in the directory
    for file_name in tqdm(os.listdir(configs.dps_raw)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(configs.dps_raw, file_name)
            logger.info(f"Cleaning {file_name}...")

            # Read CSV file into Spark DataFrame
            df = spark.read.csv(file_path, header=True, schema=configs.data_schema)

            # Ensure correct data types
            df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))

            # Sort the DataFrame by date
            df = df.sort("Date")

            # Cast columns to correct data types
            for col_name in ["Open", "High", "Low", "Close", "Adj Close"]:
                df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

            df = df.withColumn("Volume", col("Volume").cast(LongType()))

            # Check for duplicates
            df = df.dropDuplicates(["Date"])

            # Handle missing values
            # df = df.na.drop()
            df = fill_null_values(df)
            nulls_df = dump_nulls(logger, df, file_name, nulls_df)

            # Save the cleaned data back to CSV
            cleaned_file_path = os.path.join(configs.dps_clean, f"{file_name}")
            df.write.mode("overwrite").csv(cleaned_file_path, header=True)
            logger.info(f"Cleaned data saved to {cleaned_file_path}")

    # Save the results to a CSV file
    logger.info("Saving results to CSV file.")
    nulls_df.to_csv(os.path.join(configs.dps_clean, "nulls.csv"), index=False)


def process(logger):
    # Initialize Spark Session
    logger.info("Starting Spark Session...")
    spark = SparkSession.builder.appName("dps").getOrCreate()

    # Get the logger and set the logging level
    spark.sparkContext.setLogLevel("ERROR")

    clean_stock_data(spark, logger)

    # Stop the SparkSession
    logger.info("Stopping Spark Session...")
    spark.stop()
