from ray import tune
from ray.tune.schedulers import ASHAScheduler
import src.configs as configs
from ray.tune import CLIReporter
from model import StockLSTM
from functools import partial
from src.log import setup_logger, revert_streams
from optimize import train
from src.utils import mkpath
from pyspark.sql import SparkSession
from data import get_stock_metadata
import torch
from dataset import prepare_loaders


def training_function(config, metadata, spark, logger):
    # Update configs with Ray Tune's config
    configs.cyclic_loss_weight = config["cyclic_loss_weight"]
    configs.acyclic_loss_weight = 1 - config["cyclic_loss_weight"]
    configs.optimizer = config["optimizer"]
    configs.learning_rate = config["learning_rate"]
    configs.epochs = config["epochs"]
    configs.hidden_size = config["hidden_size"]
    configs.num_layers = config["num_layers"]
    configs.batch_size = config["batch_size"]

    train_loader, val_loader, test_loader = prepare_loaders(metadata, spark, logger)
    model = StockLSTM()

    # region: fix train function
    train(model, train_loader, val_loader, logger)
    # endregion


def tune_hyperparameters(
    metadata, spark, logger, num_samples=10, max_num_epochs=10, gpus_per_trial=0
):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])

    wrapped_training_function = partial(
        training_function,
        metadata=metadata,
        spark=spark,
        logger=logger,
    )

    result = tune.run(
        wrapped_training_function,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=configs.ray_tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")


if __name__ == "__main__":
    # Load and pass the logger here if needed
    logger = setup_logger(configs.ray_tune_log_name, configs.ray_tune_log_path)

    # Create Spark Session
    logger.info("Creating Spark session ...")

    # Make Spark Log Path
    mkpath(configs.rt_spark_lo_path)
    spark = (
        SparkSession.builder.appName("StockHistoryDataset")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", configs.rt_spark_lo_path)
        .config("spark.executor.extraJavaOptions", "-XX:+UseParallelGC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseParallelGC")
        .getOrCreate()
    )

    # Set Torch Seed
    torch.manual_seed(configs.torch_seed)

    # Get Stock Metadata
    metadata = get_stock_metadata(logger)

    # Call the hyperparameter tuning function with the data loaders
    tune_hyperparameters(metadata, spark, logger)

    revert_streams()
