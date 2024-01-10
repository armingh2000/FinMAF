from pathlib import Path
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
    LongType,
    BooleanType,
    IntegerType,
)

# Path of current file
current_file_path = Path(__file__).resolve()

# Project root is assumed to be some levels up
project_root = current_file_path.parent.parent.parent  # Adjust with .parent as needed
dps_raw = project_root / "data/historical/raw/"  # raw historical data path
dps_clean = project_root / "data/historical/clean/"  # clean historical data path
valid_stocks_metadata = dps_raw / "meta_data/symbols_valid_meta.csv"
# Define the schema for PySpark
data_schema = StructType(
    [
        StructField("Date", DateType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Adj Close", DoubleType(), True),
        StructField("Volume", LongType(), True),
        StructField("Repaired?", BooleanType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Month_sin", DoubleType(), True),
        StructField("Month_cos", DoubleType(), True),
        StructField("Day_sin", DoubleType(), True),
        StructField("Day_cos", DoubleType(), True),
        StructField("Open_Scaled", DoubleType(), True),
        StructField("High_Scaled", DoubleType(), True),
        StructField("Low_Scaled", DoubleType(), True),
        StructField("Close_Scaled", DoubleType(), True),
        StructField("Adj Close_Scaled", DoubleType(), True),
        StructField("Volume_Scaled", DoubleType(), True),
        StructField("Year_Scaled", DoubleType(), True),
        StructField("Row_number", LongType(), True),
    ]
)
mt_normalized = project_root / "data/historical/normalized/"
