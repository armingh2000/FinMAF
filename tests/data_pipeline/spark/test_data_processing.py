from src.data_pipeline.spark.data_processing import (
    dump_nulls,
    clean_stock_data,
    process,
)
import pandas as pd
import src.configs as configs
import os
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    DoubleType,
    LongType,
    BooleanType,
)
import pytest
from pathlib import Path


@pytest.fixture
def mock_configs(monkeypatch, tmp_path):
    # Use monkeypatch to replace yf.download with the mock function
    dps_raw = Path(tmp_path / "raw/")
    dps_raw.mkdir(exist_ok=True)
    dps_clean = Path(tmp_path / "clean/")
    dps_clean.mkdir(exist_ok=True)
    monkeypatch.setattr(configs, "dps_raw", dps_raw)
    monkeypatch.setattr(configs, "dps_clean", dps_clean)
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
    monkeypatch.setattr(configs, "data_schema", data_schema)


def test_dump_nulls(mock_logger, mock_configs, spark_session):
    nulls_df = pd.DataFrame(
        [],
        columns=["file_name", "column_with_null", "row_index"],
    )

    # Date,Open,High,Low,Close,Adj Close,Volume,Repaired?
    test_df = spark_session.createDataFrame(
        [
            ["2022-01-01", 1.0, 2.0, 3.0, 4.0, 5.0, 6, False],
            [None, 7.0, 8.0, 9.0, 10.0, 11.0, 12, True],
            ["2022-01-03", None, 14.0, 15.0, 16.0, 17.0, 18, False],
            ["2022-01-04", 19.0, None, 21.0, 22.0, 23.0, 24, True],
            ["2022-01-05", 25.0, 26.0, None, 28.0, 29.0, 30, False],
            ["2022-01-06", 31.0, 32.0, 33.0, None, 35.0, 36, True],
            ["2022-01-07", 37.0, 38.0, 39.0, 40.0, None, 42, False],
            ["2022-01-08", 43.0, 44.0, 45.0, 46.0, 47.0, None, True],
            ["2022-01-09", 48.0, 49.0, 50.0, 51.0, 52.0, 53, None],
        ],
        schema=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Repaired?",
        ],
    )

    file_name = "test_df.csv"

    nulls_df = dump_nulls(mock_logger, test_df, file_name, nulls_df)

    # Check contents of nulls_df
    # Define the expected DataFrame
    expected_data = {
        "file_name": ["test_df.csv"] * 8,
        "column_with_null": [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Repaired?",
        ],
        "row_index": [
            None,
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    assert nulls_df.equals(expected_df)


def test_clean_stock_data(mock_logger, mock_configs, spark_session, create_csv_files):
    clean_stock_data(spark_session, mock_logger)

    # Check if csv clean files are created
    assert set(["test_data1.csv", "test_data2.csv"]).issubset(
        os.listdir(configs.dps_clean)
    )

    # Check for nulls.csv creation
    assert os.path.exists(configs.dps_clean / "nulls.csv")

    # Define the schema
    schema = configs.data_schema

    # Check for removal of duplicates
    test_df1_path = os.path.join(configs.dps_clean, "test_data1.csv")
    test_df2_path = os.path.join(configs.dps_clean, "test_data2.csv")
    test_df1 = spark_session.read.csv(test_df1_path, header=True, schema=schema)
    test_df2 = spark_session.read.csv(test_df2_path, header=True, schema=schema)

    duplicate_dates1 = test_df1.groupBy("date").count().filter(col("count") > 1)
    duplicate_dates2 = test_df2.groupBy("date").count().filter(col("count") > 1)

    assert duplicate_dates1.count() == 0
    assert duplicate_dates2.count() == 0

    # Check for data types
    assert test_df1.dtypes == [
        ("Date", "date"),
        ("Open", "double"),
        ("High", "double"),
        ("Low", "double"),
        ("Close", "double"),
        ("Adj close", "double"),
        ("Volume", "bigint"),
        ("Repaired?", "boolean"),
    ]

    assert test_df2.dtypes == [
        ("Date", "date"),
        ("Open", "double"),
        ("High", "double"),
        ("Low", "double"),
        ("Close", "double"),
        ("Adj close", "double"),
        ("Volume", "bigint"),
        ("Repaired?", "boolean"),
    ]

    # Check for contents of nulls.csv
    nulls_df = pd.read_csv(configs.dps_clean / "nulls.csv")

    assert set(["test_data1.csv", "test_data2.csv"]).issubset(
        list(nulls_df["file_name"])
    )


def test_process(mock_logger, mock_spark_session, mock_clean_stock_data):
    # Call the process function
    process(mock_logger)

    # Assert that the appName method was called on the builder
    mock_spark_session.appName.assert_called_once_with("dps")

    # Assert that getOrCreate was called
    mock_spark_session.getOrCreate.assert_called_once()

    # Assert that clean_stock_data was called correctly
    mock_clean_stock_data.assert_called_once_with(mock_spark_session, mock_logger)
