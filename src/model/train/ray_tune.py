from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig
from ray import train, tune
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
from optimize import train as train_func
from optimize import evaluate as evaluate_func


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

    train_loader, val_loader, _ = prepare_loaders(metadata, spark, logger)
    model = StockLSTM()

    # region: fix train function
    train_func(model, train_loader, val_loader, logger)
    # endregion


def tune_hyperparameters(
    metadata, spark, logger, num_samples=10, max_num_epochs=10, gpus_per_trial=0
):
    # Initialize the scheduler.
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    # Initialize the reporter.
    reporter = CLIReporter(
        metric_columns=[
            "epoch loss",
            "mid-train loss",
            "iteration",
        ]
    )

    # Initialize the search algorithm.
    search_alg = HyperOptSearch()

    wrapped_training_function = partial(
        training_function,
        metadata=metadata,
        spark=spark,
        logger=logger,
    )

    tuner = tune.Tuner(
        wrapped_training_function,
        tune_config=tune.TuneConfig(
            num_samples=10,
            search_alg=search_alg,
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        ),
        param_space=configs.ray_tune_config,
    )

    results = tuner.fit()

    best_trial = results.get_best_trial("loss", "min", "last")
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
