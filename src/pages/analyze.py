import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from src.model.model import FinMAF
import src.configs as configs
import torch
from src.model.predict import predict


@st.cache_resource
def get_model():
    model = FinMAF(configs.data_dim)
    print("Loading model weights...")
    model.load_state_dict(torch.load(configs.model_path))

    return model


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


def get_preds(symbol):
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


if "symbol" not in st.session_state:
    st.switch_page(configs.input_form_page)

symbol = st.session_state["symbol"]

actual_candlestick = get_actuals(symbol)

model = get_model()
pred_candlestick = get_preds(symbol)

fig = go.Figure(
    data=[actual_candlestick, pred_candlestick],
)

st.plotly_chart(fig)
