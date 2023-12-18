from pathlib import Path
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
    LongType,
    BooleanType,
)

# Path of current file
current_file_path = Path(__file__).resolve()

# Project root is assumed to be some levels up
project_root = current_file_path.parent.parent.parent  # Adjust with .parent as needed
dps_raw = project_root / "data/historical/raw/"  # raw historical data path
dps_clean = project_root / "data/historical/clean/"  # clean historical data path
# Define the schema
data_schema = StructType(
    [
        StructField("Date", DateType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("Adj close", DoubleType(), True),
        StructField("Volume", LongType(), True),
        StructField("Repaired?", BooleanType(), True),
    ]
)
