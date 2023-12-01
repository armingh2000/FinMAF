from pathlib import Path

# Path of current file
current_file_path = Path(__file__).resolve()

# Project root is assumed to be some levels up
project_root = current_file_path.parent.parent.parent  # Adjust with .parent as needed
dps_raw = project_root / "data/historical/raw/"  # raw historical data path
dps_clean = project_root / "data/historical/clean/"  # clean historical data path
