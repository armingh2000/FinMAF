import src.configs as configs
from src.utils import mkpath


def make_paths():
    mkpath(configs.model_path)
