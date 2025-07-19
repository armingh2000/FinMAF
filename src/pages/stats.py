import src.configs as configs
from src.model.predict import predict
import yfinance as yf
import plotly.graph_objects as go
import src.patterns.candlestick as candlestick


def get_actuals(symbol):
    df = yf.download(tickers=symbol, period="3mo", interval="1d", repair=True)

    df = df[-configs.window_size * 2 :]

    actual_candlestick = go.Candlestick(
        x=df.index,
        open=df["Open"][symbol],
        high=df["High"][symbol],
        low=df["Low"][symbol],
        close=df["Close"][symbol],
        name=f"{symbol} Actual",
    )

    return actual_candlestick


def get_preds(symbol, model):
    pred = predict(model, symbol, "1m")

    pred_candlestick = go.Candlestick(
        x=pred.index,
        open=pred["Open"],
        high=pred["High"],
        low=pred["Low"],
        close=pred["Close"],
        name=f"{symbol} Prediction",
    )

    pred_candlestick.increasing.line.color = "ghostwhite"
    pred_candlestick.decreasing.line.color = "grey"

    return pred_candlestick


def find_inverted_hammer_positions(df):
    result = candlestick.inverted_hammer(df, target="result")
    inverted_hammer_positions = df[result["result"]]

    return inverted_hammer_positions.index, inverted_hammer_positions["Low"]
