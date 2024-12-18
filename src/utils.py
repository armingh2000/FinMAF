import os
from sklego.preprocessing import RepeatingBasisFunction
from sklearn.preprocessing import RobustScaler
import pandas_market_calendars as mcal
import pandas as pd


def mkpath(path):
    """
    Checks if the given path is a file or a directory and creates it if it doesn't exist.

    :param path: str - The file or directory path to check and create.
    """
    if not os.path.exists(path):
        if "." in os.path.basename(
            path
        ):  # Assuming it's a file if there's an extension
            # Create the parent directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Create an empty file
            with open(path, "w") as f:
                pass
        else:
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")


def get_scalers(df):
    df["day_of_year"] = df.index.dayofyear

    rbf = RepeatingBasisFunction(
        n_periods=12, column="day_of_year", input_range=(1, 365)
    )
    rs_p = RobustScaler()
    rs_v = RobustScaler()

    rbf.fit(df)
    rs_p.fit(df[["Close", "High", "Low", "Open"]])
    rs_v.fit(df[["Volume"]])

    return rbf, rs_p, rs_v


def scale_df(df, for_prediction):

    if for_prediction:
        rbf, rs_p, rs_v = get_scalers(df)
        df["day_of_year"] = df.index.dayofyear
        transformed_features = rbf.transform(df)
        for i in range(transformed_features.shape[1]):
            df[f"rbf_{i}"] = transformed_features[:, i]

    else:
        rbf, rs_p, rs_v = get_scalers(df[:-1])

    df[["Close_Scaled", "High_Scaled", "Low_Scaled", "Open_Scaled"]] = rs_p.transform(
        df[["Close", "High", "Low", "Open"]]
    )

    df["Volume_Scaled"] = rs_v.transform(df[["Volume"]])

    return df if not for_prediction else df, rs_p, rs_v


def extract_filename_without_extension(file_path):
    """
    Extracts the filename (without extension) from a file path.

    Args:
        file_path: The path to the file (string).

    Returns:
        The filename without extension (string) or None if the path is invalid or doesn't represent a file.
    """
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)  # Split filename and extension

    return name


def next_n_trading_days(n, last_trading_day=None):
    """
    Returns the next N trading days for the NASDAQ stock market.

    Args:
        n: The number of trading days to find.

    Returns:
        A list of datetime.date objects representing the next N trading days.
    """

    # Get the current date
    today = (
        pd.to_datetime("today").to_datetime64()
        if last_trading_day is None
        else last_trading_day
    )

    # Create a market calendar for the NASDAQ
    nasdaq = mcal.get_calendar("NASDAQ")

    # Find the next N trading days
    start_date = today - pd.DateOffset(days=3)  # Adjust start date for buffer
    end_date = today + pd.DateOffset(days=n * 2)  # Adjust end date for buffer
    trading_days = nasdaq.schedule(start_date=start_date, end_date=end_date).index

    # Filter for trading days after today
    next_trading_days = trading_days[trading_days > today][:n]

    return next_trading_days
