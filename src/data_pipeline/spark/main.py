from data_download import download_symbols, save_symbols
from src.log import revert_streams, setup_logger
import src.configs as configs
from data_processing import process
from utils import make_paths

if __name__ == "__main__":
    make_paths()

    logger = setup_logger(configs.dps_download_log_name, configs.dps_download_log_path)
    symbols, df = download_symbols(logger)
    valid_data = save_symbols(symbols, df, logger)

    logger = setup_logger(configs.dps_process_log_name, configs.dps_process_log_path)

    process(logger)

    revert_streams()
