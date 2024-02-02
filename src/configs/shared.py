from pathlib import Path
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
    LongType,
    BooleanType,
)
import platform

# Check if running on Windows
is_windows = platform.system() == "Windows"

# Windows reserved device names
windows_reserved_names = [
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
]

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
        StructField("Adj close", DoubleType(), True),
        StructField("Volume", LongType(), True),
        StructField("Repaired?", BooleanType(), True),
    ]
)
