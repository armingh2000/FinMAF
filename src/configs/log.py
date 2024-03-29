from .shared import project_root

# Data Pipeline Logging Configurations
dps_download_log_name = "dps_download"
dps_download_log_path = project_root / "logs/dps_download.log"
dps_process_log_name = "dps_process"
dps_process_log_path = project_root / "logs/dps_process.log"

# Model Train Logging Configurations
embedding_log_name = "mt_embedding"
embedding_log_path = project_root / "logs/mt_embedding.log"
