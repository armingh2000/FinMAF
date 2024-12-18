from src.data.download import download_symbols, save_symbols
from src.data.utils import make_paths
from src.data.process import load_raw_symbols, process_symbols

if __name__ == "__main__":
    make_paths()

    # symbols, df = download_symbols()
    # valid_data = save_symbols(symbols, df)

    symbols = load_raw_symbols()
    process_symbols(symbols)
