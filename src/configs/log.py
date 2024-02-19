from .shared import project_root

# Data Pipeline Logging Configurations
dps_download_log_name = "dps_download"
dps_download_log_path = project_root / "logs/dps_download.log"
dps_process_log_name = "dps_process"
dps_process_log_path = project_root / "logs/dps_process.log"

# Model Optimize Logging Configurations
embedding_log_name = "mt_embedding"
embedding_log_path = project_root / "logs/mt_embedding.log"
normalize_log_name = "mt_normalize"
normalize_log_path = project_root / "logs/mt_normalize.log"
stock_history_dataset_log_name = "mt_stock_history_dataset"
stock_history_dataset_log_path = project_root / "logs/mt_stock_history_dataset.log"
mt_spark_log_path = project_root / "logs/spark/model_train/"
optimize_log_path = project_root / "logs/mt_optimize.log"
optimize_log_name = "mt_optimize"

## Ray Tune
ray_tune_log_path = project_root / "logs/ray_tune/"
ray_tune_log_name = "mt_ray_tune.log"
