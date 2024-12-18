from pathlib import Path

# Path of current file
current_file_path = Path(__file__).resolve()

# Project root is assumed to be some levels up
project_root = current_file_path.parent.parent.parent  # Adjust with .parent as needed
data_raw = project_root / "data/historical/raw/"  # raw historical data path
valid_stocks_metadata = data_raw / "meta_data/symbols_valid_meta.csv"
failed_stocks_metadata = data_raw / "meta_data/symbols_failed_meta.csv"
data_processed = project_root / "data/historical/processed/"
model_path = project_root / "models/model.pth"
dataset_path = project_root / "data/historical/dataset.pt"
