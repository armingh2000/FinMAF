from src.data_pipeline.spark.data_download import download_symbols, save_symbols
from unittest.mock import Mock
import pandas as pd
import yfinance as yf
from pathlib import Path
import pytest
import src.configs as configs


def test_successful_data_retrieval_and_filtering(monkeypatch, logger):
    # Mock pd.read_csv to return a predefined DataFrame
    test_data = pd.DataFrame(
        {
            "Test Issue": ["N", "Y", "N", "N"],
            "NASDAQ Symbol": ["AAPL", "GOOGL", "MSFT$", "AMZN"],
        }
    )
    monkeypatch.setattr(pd, "read_csv", Mock(return_value=test_data))

    logger = Mock()
    symbols, _ = download_symbols(logger)

    assert isinstance(symbols, list), "Returned symbols should be a list"
    assert all(isinstance(sym, str) for sym in symbols), "All symbols should be strings"
    assert "GOOGL" not in symbols, "Test issues should be filtered out"
    assert "MSFT$" not in symbols, "Non-alphabetic symbols should be filtered out"


def test_correct_handling_of_non_alphabetic_symbols(monkeypatch, logger):
    # Mock pd.read_csv
    test_data = pd.DataFrame(
        {
            "Test Issue": ["N", "N", "N", "N"],
            "NASDAQ Symbol": ["AAPL", "1234", "MSFT$", "AMZN"],
        }
    )
    monkeypatch.setattr(pd, "read_csv", Mock(return_value=test_data))

    logger = Mock()
    _, data_clean = download_symbols(logger)

    assert all(
        data_clean["NASDAQ Symbol"].apply(lambda x: x.isalpha())
    ), "All NASDAQ Symbols should be strictly alphabetic"


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


@pytest.fixture
def mock_configs(monkeypatch, tmp_path):
    # Use monkeypatch to replace yf.download with the mock function
    dps_raw = Path(tmp_path / "raw/")
    dps_raw.mkdir()
    meta_file_path = Path(tmp_path / "raw/meta_data/")
    meta_file_path.mkdir()
    monkeypatch.setattr(configs, "limit", 3)
    monkeypatch.setattr(configs, "offset", 1)
    monkeypatch.setattr(configs, "dps_raw", dps_raw)
    monkeypatch.setattr(configs, "meta_file_path", meta_file_path)


def test_save_symbols(symbols, logger, data_clean, mock_configs, mock_yf_download):
    valid_df = save_symbols(symbols, data_clean, logger)

    # df symbols
    assert "AB" in valid_df["Symbol"].values
    assert "ABEO" in valid_df["Symbol"].values

    # limit
    assert len(valid_df) <= configs.limit

    # offset
    assert "AA" not in valid_df["Symbol"]

    # check failed_df
    failed_df = pd.read_csv(configs.meta_file_path / "symbols_failed_meta.csv")
    assert len(failed_df) == 1
    assert "ACONW" in failed_df["Symbol"].values

    # check int casting of float volumes
    for csv_file in configs.dps_raw.glob("*.csv"):
        df = pd.read_csv(csv_file)
        assert df["Volume"].dtype == "int64"
