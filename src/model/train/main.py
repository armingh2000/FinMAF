from data import get_stock_metadata, dump_normalized_dataset
from embedding import get_embeddings, get_embedding_input, perform_pca
from utils import dump_embeddings, load_embeddings
import src.configs as configs
from src.log import setup_logger, revert_streams
from dataset import StockHistoryDataset
from pyspark.sql import SparkSession

if __name__ == "__main__":
    # setup logger
    logger = setup_logger(configs.embedding_log_name, configs.embedding_log_path)

    # get stock metadata
    metadata = get_stock_metadata(logger)

    # # get embeddings input for bert
    # embedding_inputs = get_embedding_input(metadata, logger)

    # # get embeddings of stocks
    # stock_embeddings = get_embeddings(embedding_inputs, logger)

    # # dump embeddings
    # logger.info("Dumping embeddings ...")
    # dump_embeddings(stock_embeddings, configs.bert_embedding_path)

    # load embeddings
    # logger.info("Loading embeddings ...")
    # stock_embeddings = load_embeddings(configs.bert_embedding_path)

    # Getting PCA embeddings
    # pca_embeddings = perform_pca(stock_embeddings, logger, configs.n_pca_components)

    # Dump PCA embeddings
    # logger.info("Dumping PCA embeddings ...")
    # dump_embeddings(pca_embeddings, configs.pca_embedding_path)

    # load embeddings
    # logger.info("Loading PCA embeddings ...")
    # pca_embeddings = load_embeddings(configs.pca_embedding_path)

    logger = setup_logger(configs.normalize_log_name, configs.normalize_log_path)

    # Creating Spark Session
    logger.info("Creating Spark session ...")
    spark = (
        SparkSession.builder.appName("StockHistoryDataset")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", configs.mt_spark_log_path)
        .config("spark.executor.extraJavaOptions", "-XX:+UseParallelGC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseParallelGC")
        .getOrCreate()
    )

    # Saving normalized dataset to CSV files
    logger.info("Dumping normalized dataset ...")
    dump_normalized_dataset(metadata, spark, logger)

    # dataset = StockHistoryDataset(metadata, pca_embeddings, logger)

    # revert std streams
    revert_streams()
