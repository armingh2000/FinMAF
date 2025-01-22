import streamlit as st
import src.configs as configs
from src.pages.cache import get_model
from src.pages.stats import get_actuals, get_preds, find_inverted_hammer_positions
import plotly.graph_objects as go
from src.model.predict import download_symbol


if "symbol" not in st.session_state:
    st.switch_page(configs.input_form_page)

symbol = st.session_state["symbol"]

model = get_model()
actual_candlestick = get_actuals(symbol)
pred_candlestick = get_preds(symbol, model)

df = download_symbol(symbol)[-configs.window_size :]
x_hammer, y_hammer = find_inverted_hammer_positions(df)

fig = go.Figure(
    data=[actual_candlestick, pred_candlestick],
)

fig.add_trace(
    go.Scatter(
        x=x_hammer,
        y=y_hammer,
        mode="markers",
        marker=dict(size=10, color="green", symbol="arrow-up"),
        name="Inverted Hammer",
    )
)

st.plotly_chart(fig)
