import pytest
import os
import csv
from pyspark.sql import SparkSession
from pathlib import Path
import src.configs as configs
import pytest
from unittest.mock import Mock
import shutil


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
def mock_configs(monkeypatch, tmp_path):
    # Use monkeypatch to replace yf.download with the mock function
    dps_raw = Path(tmp_path / "raw/")
    dps_raw.mkdir(exist_ok=True)
    dps_clean = Path(tmp_path / "clean/")
    dps_clean.mkdir(exist_ok=True)
    monkeypatch.setattr(configs, "dps_raw", dps_raw)
    monkeypatch.setattr(configs, "dps_clean", dps_clean)


@pytest.fixture
def mock_logger():
    return Mock()
