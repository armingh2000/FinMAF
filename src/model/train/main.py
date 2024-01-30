from data import get_stock_metadata, dump_normalized_data
from embedding import get_embeddings, get_embedding_input, perform_pca
from src.utils import dump_dictionary, load_dictionary
import src.configs as configs
from src.log import setup_logger, revert_streams
from pyspark.sql import SparkSession
from src.utils import mkpath
from torch.utils.data import DataLoader
from dataset import prepare_loaders, StockHistoryDataset
from model import StockLSTM
from optimize import train
import torch

if __name__ == "__main__":
    # setup logger
    logger = setup_logger(configs.embedding_log_name, configs.embedding_log_path)

    # get stock metadata
    metadata = get_stock_metadata(logger)

    # get embeddings input for bert
    # embedding_inputs = get_embedding_input(metadata, logger)

    # get embeddings of stocks
    # stock_embeddings = get_embeddings(embedding_inputs, logger)

    # dump embeddings
    # logger.info("Dumping embeddings ...")
    # dump_dictionary(stock_embeddings, configs.bert_embedding_path)

    # load embeddings
    # logger.info("Loading embeddings ...")
    # stock_embeddings = load_dictionary(configs.bert_embedding_path)

    # Getting PCA embeddings
    # pca_embeddings = perform_pca(stock_embeddings, logger)

    # Dump PCA embeddings
    # logger.info("Dumping PCA embeddings ...")
    # dump_dictionary(pca_embeddings, configs.pca_embedding_path)

    # load embeddings
    # logger.info("Loading PCA embeddings ...")
    # pca_embeddings = load_dictionary(configs.pca_embedding_path)

    # logger = setup_logger(configs.normalize_log_name, configs.normalize_log_path)

    # Creating Spark Session
    # logger.info("Creating Spark session ...")
    # Make spark log path
    mkpath(configs.mt_spark_log_path)
    spark = (
        SparkSession.builder.appName("StockHistoryDataset")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", configs.mt_spark_log_path)
        .config("spark.executor.extraJavaOptions", "-XX:+UseParallelGC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseParallelGC")
        .getOrCreate()
    )

    # Saving normalized dataset to CSV files
    # logger.info("Dumping normalized dataset ...")
    # dump_normalized_data(metadata, spark, logger)

    logger = setup_logger(
        configs.stock_history_dataset_log_name, configs.stock_history_dataset_log_path
    )

    # Setting torch seed
    torch.manual_seed(configs.torch_seed)

    train_loader, val_loader, test_loader = prepare_loaders(metadata, spark, logger)

    model = StockLSTM()

    train(model, train_loader, val_loader, logger)

    # revert std streams
    revert_streams()
