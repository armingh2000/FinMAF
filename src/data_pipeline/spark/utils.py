import shutil
from os.path import join
import src.configs as configs
from src.utils import mkpath


def separate_ETFs_and_Stocks(valid_data):
    etfs = valid_data[valid_data["ETF"] == "Y"]["Symbol"].tolist()
    stocks = valid_data[valid_data["ETF"] == "N"]["Symbol"].tolist()

    move_symbols(etfs, configs.dps_raw / "etfs")
    move_symbols(stocks, configs.dps_raw / "stocks")


def move_symbols(symbols, dest):
    for s in symbols:
        filename = "{}.csv".format(s)
        shutil.move(join(configs.dps_raw / "hist", filename), join(dest, filename))


def make_paths():
    mkpath(configs.dps_raw)
    mkpath(configs.dps_clean)
    mkpath(configs.meta_file_path)
