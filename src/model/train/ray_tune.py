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
from optimize import train_func


def training_function(config, metadata, spark, logger):
    config["acyclic_loss_weight"] = 1 - config["cyclic_loss_weight"]

    train_loader, val_loader, _ = prepare_loaders(metadata, spark, logger)
    model = StockLSTM(config["hidden_size"], config["num_layers"])

    train_func(model, config, train_loader, val_loader, logger)


def tune_hyperparameters(
    metadata,
    spark,
    logger,
    max_num_epochs=10,
):
    # Initialize the scheduler.
    scheduler = ASHAScheduler(
        metric="epoch loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    # Initialize the reporter.
    reporter = CLIReporter(
        metric_columns=[
            "epoch",
            "epoch loss",
            "epoch accuracy",
            "iteration",
            "iteration loss",
            "iteration accuracy",
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
            num_samples=configs.rt_num_samples,
            search_alg=search_alg,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name=configs.ray_tune_exp_name,
            storage_path=configs.ray_tune_exp_path,
            progress_reporter=reporter,
            resources_per_trial=configs.rt_resources_per_trial,
        ),
        param_space=configs.rt_config,
    )

    result_grid = tuner.fit()


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
