import pytest
import os
import csv
from pyspark.sql import SparkSession
from pathlib import Path
import src.configs as configs
from unittest.mock import Mock
import shutil
import pandas as pd
import yfinance as yf


@pytest.fixture(scope="module")
def logger():
    return Mock()


@pytest.fixture(scope="module")
def symbols():
    symbols = ["AA", "ABEO", "AB", "ACONW", "AC"]  # ACONW should fail
    return symbols


@pytest.fixture(scope="module")
def data_clean(symbols):
    return pd.DataFrame(
        {
            "Symbol": symbols,
        }
    )


@pytest.fixture()
def mock_yf_download(monkeypatch):
    # Define a side effect function for yf.download
    def yf_download_side_effect(symbol, *args, **kwargs):
        if symbol == "AA":
            # Return a DataFrame for AAPL
            return pd.DataFrame({"Volume": [1000, 2000, 1500]})
        elif symbol == "AB":
            # Return a DataFrame for MSFT
            return pd.DataFrame({"Volume": [3000.3, 4000, 3500]})
        elif symbol == "ABEO":
            # Return a DataFrame for MSFT
            return pd.DataFrame({"Volume": [300, 4000.4, 3500]})
        elif symbol == "AC":
            # Return a DataFrame for MSFT
            return pd.DataFrame({"Volume": [3000, 4000, 3500]})
        else:
            # Return an empty DataFrame for other symbols
            return pd.DataFrame()

    # Use monkeypatch to replace yf.download with the mock function
    monkeypatch.setattr(yf, "download", Mock(side_effect=yf_download_side_effect))


@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.appName("test").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def clean_tmp_path(tmp_path):
    # Cleanup code to manually clear the tmp_path
    for item in tmp_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


@pytest.fixture
def create_csv_files(tmp_path, clean_tmp_path, mock_configs, mkdirs):
    # Define CSV files and their content
    files_content = {
        "test_data1.csv": [
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Repaired?",
            ],
            ["2022-01-01", 1.0, 2.0, 3.0, 4.0, 5.0, 6, False],
            [None, 7.0, 8.0, 9.0, 10.0, 11.0, 12, True],
            ["2022-01-03", None, 14.0, 15.0, 16.0, 17.0, 18, False],
            ["2022-01-04", 19.0, None, 21.0, 22.0, 23.0, 24, True],
            ["2022-01-05", 25.0, 26.0, None, 28.0, 29.0, 30, False],
            ["2022-01-06", 31.0, 32.0, 33.0, None, 35.0, 36, True],
            ["2022-01-06", 31.0, 32.0, 33.0, None, 35.0, 36, True],  # DUPLICATE
        ],
        "test_data2.csv": [
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Repaired?",
            ],
            ["2022-01-01", 1.0, 2.0, 3.0, 4.0, 5.0, 6, False],
            ["2022-01-02", 37.0, 38.0, 39.0, 40.0, None, 42, False],
            ["2022-01-03", 43.0, 44.0, 45.0, 46.0, 47.0, None, True],
            ["2022-01-04", 48.0, 49.0, 50.0, 51.0, 52.0, 53, None],
            ["2022-01-04", 48.0, 49.0, 50.0, 51.0, 52.0, 53, None],  # DUPLICATE
        ],
    }

    # Create CSV files
    for file_name, content in files_content.items():
        file_path = os.path.join(configs.dps_raw, file_name)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(content)


@pytest.fixture
def mkdirs(mock_configs):
    configs.dps_raw.mkdir(exist_ok=True)
    configs.dps_clean.mkdir(exist_ok=True)


@pytest.fixture
def mock_logger():
    return Mock()


# Mock for SparkSession
@pytest.fixture
def mock_spark_session(monkeypatch):
    # Create mocks for SparkSession and its builder
    mock_builder = Mock()
    mock_builder.appName.return_value = mock_builder
    mock_builder.getOrCreate.return_value = mock_builder

    # Patch the SparkSession builder
    monkeypatch.setattr("pyspark.sql.SparkSession.builder", mock_builder)

    return mock_builder


# Mock for clean_stock_data
@pytest.fixture
def mock_clean_stock_data(monkeypatch):
    # Mock clean_stock_data function
    mock_function = Mock()
    monkeypatch.setattr(
        "src.data_pipeline.spark.data_processing.clean_stock_data",
        mock_function,
    )
    return mock_function
