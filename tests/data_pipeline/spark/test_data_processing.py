from src.data_pipeline.spark.data_processing import (
    dump_nulls,
    clean_stock_data,
    process,
)
from unittest.mock import Mock
import pandas as pd
from pathlib import Path
import pytest
import src.configs as configs


@pytest.fixture(scope="module")
def mock_logger():
    return Mock()


@pytest.fixture
def mock_configs(monkeypatch, tmp_path):
    # Use monkeypatch to replace yf.download with the mock function
    dps_raw = Path(tmp_path / "raw/")
    dps_raw.mkdir()
    dps_clean = Path(tmp_path / "clean/")
    dps_clean.mkdir()
    monkeypatch.setattr(configs, "dps_raw", dps_raw)
    monkeypatch.setattr(configs, "dps_clean", dps_clean)


def test_dump_nulls(mock_logger, mock_configs, spark_session):
    nulls_df = pd.DataFrame(
        [],
        columns=["file_name", "column_with_null", "row_index"],
    )

    # Date,Open,High,Low,Close,Adj Close,Volume,Repaired?
    test_df = spark_session.createDataFrame(
        [
            ["2022-01-01", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, False],
            [None, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, True],
            ["2022-01-03", None, 14.0, 15.0, 16.0, 17.0, 18.0, False],
            ["2022-01-04", 19.0, None, 21.0, 22.0, 23.0, 24.0, True],
            ["2022-01-05", 25.0, 26.0, None, 28.0, 29.0, 30.0, False],
            ["2022-01-06", 31.0, 32.0, 33.0, None, 35.0, 36.0, True],
            ["2022-01-07", 37.0, 38.0, 39.0, 40.0, None, 42.0, False],
            ["2022-01-08", 43.0, 44.0, 45.0, 46.0, 47.0, None, True],
            ["2022-01-09", 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, None],
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
