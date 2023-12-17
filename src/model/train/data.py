import pandas as pd
import src.configs as configs


def get_stock_metadata(logger):
    logger.info("Getting valid stock metadata ...")

    metadata = pd.read_csv(
        configs.valid_stocks_metadata,
        dtype={"Symbol": str, "NASDAQ Symbol": str},
        keep_default_na=False,  # This prevents Pandas from interpreting "NA" as NaN
    )

    return metadata
