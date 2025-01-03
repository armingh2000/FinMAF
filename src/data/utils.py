import src.configs as configs
from src.utils import mkpath


def make_paths():
    mkpath(configs.data_raw)
    mkpath(configs.metadata_file_path)
    mkpath(configs.data_processed)
