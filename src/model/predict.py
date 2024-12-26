import yfinance as yf
import torch
import src.configs as configs
from src.model.model import *
from src.utils import *
import numpy as np
import pandas as pd


def download_symbol(symbol):
    df = yf.download(symbol, period="6mo", interval="1d", repair=True)

    # Check for corrupted data (non-integer values)
    df.columns = df.columns.droplevel(1)
    df.columns = ["Close", "High", "Low", "Open", "Repaired?", "Volume"]
    corrupted_indices = df["Volume"].apply(lambda x: x != int(x))

    # Strategy 3: Round to nearest integer
    df.loc[corrupted_indices, "Volume"] = df.loc[corrupted_indices, "Volume"].round()
    df["Volume"] = df["Volume"].astype("int64")

    return df


def get_input_tensor(df):
    df_copy = df.copy()
    df_copy = df_copy.iloc[-configs.window_size :]
    input_df = df_copy[
        [
            "Close_Scaled",
            "High_Scaled",
            "Low_Scaled",
            "Open_Scaled",
            "Volume_Scaled",
            "rbf_0",
            "rbf_1",
            "rbf_2",
            "rbf_3",
            "rbf_4",
            "rbf_5",
            "rbf_6",
            "rbf_7",
            "rbf_8",
            "rbf_9",
            "rbf_10",
            "rbf_11",
        ]
    ]

    input_tensor = torch.FloatTensor(input_df.to_numpy()).unsqueeze(0)

    return input_tensor


def predict_future(model, symbol, period):
    preds = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    preds.set_index("Date", inplace=True)

    df = download_symbol(symbol)

    n_days = {
        "1d": 1,
        "1w": 7,
        "1m": 30,
    }[period]
    next_trading_days = next_n_trading_days(n_days)

    for i in range(n_days):
        df = df[-configs.window_size :]
        df, rs_p, rs_v = scale_df(df, True)

        input_tensor = get_input_tensor(df)

        pred = model(input_tensor, False)

        close_pred, high_pred, low_pred, open_pred, volume_pred = (
            pred[0].detach().numpy()
        )
        price_array = np.array([close_pred, high_pred, low_pred, open_pred])
        volume_array = np.array([volume_pred])

        price_df = pd.DataFrame([price_array], columns=["Close", "High", "Low", "Open"])
        volume_df = pd.DataFrame([volume_array], columns=["Volume"])

        price_df[["Close", "High", "Low", "Open"]] = rs_p.inverse_transform(
            price_df[["Close", "High", "Low", "Open"]]
        )[0]
        volume_df["Volume"] = rs_v.inverse_transform(volume_df[["Volume"]])[0]

        price_array = price_df.values[0]
        volume_array = volume_df.values[0]
        open_actual, high_actual, low_actual, close_actual = price_array
        volume_actual = volume_array[0].astype(int)

        df.loc[next_trading_days[i]] = {
            "Open": open_actual,
            "High": high_actual,
            "Low": low_actual,
            "Close": close_actual,
            "Volume": volume_actual,
        }

        preds.loc[next_trading_days[i]] = {
            "Open": open_actual,
            "High": high_actual,
            "Low": low_actual,
            "Close": close_actual,
            "Volume": volume_actual,
        }

    return preds


def predict_history(model, symbol):
    preds = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    preds.set_index("Date", inplace=True)

    df_org = download_symbol(symbol)

    n_days = 30
    next_trading_days = next_n_trading_days(n_days, df_org.index[-n_days])

    for i in range(n_days):
        df = df_org.copy()[-configs.window_size - n_days + i : -n_days + i]

        df, rs_p, rs_v = scale_df(df, True)

        input_tensor = get_input_tensor(df)

        pred = model(input_tensor, False)

        close_pred, high_pred, low_pred, open_pred, volume_pred = (
            pred[0].detach().numpy()
        )
        price_array = np.array([close_pred, high_pred, low_pred, open_pred])
        volume_array = np.array([volume_pred])

        price_df = pd.DataFrame([price_array], columns=["Close", "High", "Low", "Open"])
        volume_df = pd.DataFrame([volume_array], columns=["Volume"])

        price_df[["Close", "High", "Low", "Open"]] = rs_p.inverse_transform(
            price_df[["Close", "High", "Low", "Open"]]
        )[0]
        volume_df["Volume"] = rs_v.inverse_transform(volume_df[["Volume"]])[0]

        price_array = price_df.values[0]
        volume_array = volume_df.values[0]
        open_actual, high_actual, low_actual, close_actual = price_array
        volume_actual = volume_array[0].astype(int)

        preds.loc[next_trading_days[i]] = {
            "Open": open_actual,
            "High": high_actual,
            "Low": low_actual,
            "Close": close_actual,
            "Volume": volume_actual,
        }

    return preds, df


def predict(model, symbol, period):
    pred_future = predict_future(model, symbol, period)[1:]
    pred_history, _ = predict_history(model, symbol)

    pred = pd.concat([pred_history, pred_future], axis=0).sort_index()

    return pred


if __name__ == "__main__":
    model = FinMAF(configs.data_dim)
    print("Loading model weights...")
    model.load_state_dict(torch.load(configs.model_path))
    symbol = "AMZN"
    pred_future = predict_future(model, symbol, "1m")
    pred_history, df = predict_history(model, symbol)
    # print(pred_future)
    print(df[-30:][["Open", "High", "Low", "Close", "Volume"]])
    print(pred_history)
