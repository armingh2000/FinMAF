from torch.utils.data import Dataset
import src.configs as configs
from data import dump_stock_durations
from src.utils import load_dictionary


class StockHistoryDataset(Dataset):
    def __init__(self, metadata, spark, logger):
        self.metadata = metadata
        self.spark = spark
        self.logger = logger
        self.lstm_sequence_length = configs.lstm_sequence_length
        self.durations = self.load_durations()
        if not self.durations:
            self.logger.info("No duration data found")
            self.durations = dump_stock_durations(metadata, spark, logger)
        self.length = self.calculate_length()

    def load_durations(self):
        try:
            duration = load_dictionary(
                configs.stock_durations_path, self.logger, "durations"
            )

        except ValueError:
            duration = None

        return duration

    def calculate_length(self):
        pass

    def __len__(self):
        self.length

    def __getitem__(self, idx):
        pass
