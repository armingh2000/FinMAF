import streamlit as st
import src.configs as configs
from src.pages.cache import get_model
from src.pages.stats import get_actuals, get_preds, find_inverted_hammer_positions
import plotly.graph_objects as go
from src.model.predict import download_symbol


if "symbol" not in st.session_state:
    st.switch_page(configs.input_form_page)

col1, col2 = st.columns([3, 1])

symbol = st.session_state["symbol"]

model = get_model()

if "actual_candlestick" not in st.session_state:
    st.session_state.actual_candlestick = get_actuals(symbol)

if "pred_candlestick" not in st.session_state:
    st.session_state.pred_candlestick = get_preds(symbol, model)

if "df" not in st.session_state:
    st.session_state.df = download_symbol(symbol)[-configs.window_size :]
    st.session_state.x_hammer, st.session_state.y_hammer = (
        find_inverted_hammer_positions(st.session_state.df)
    )

if "show_inverted_hammer" not in st.session_state:
    st.session_state.show_inverted_hammer = False

with col2:
    if st.button("Find Inverted Hammer"):
        st.session_state.show_inverted_hammer = (
            not st.session_state.show_inverted_hammer
        )

with col1:
    fig = go.Figure(
        data=[st.session_state.actual_candlestick, st.session_state.pred_candlestick],
    )

    if st.session_state.show_inverted_hammer:
        fig.add_trace(
            go.Scatter(
                x=st.session_state.x_hammer,
                y=0.97 * st.session_state.y_hammer,
                mode="markers",
                marker=dict(size=10, color="green", symbol="arrow-up"),
                name="Inverted Hammer",
            )
        )

    st.plotly_chart(fig)
