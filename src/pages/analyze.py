import streamlit as st
import src.configs as configs
from src.pages.cache import get_model
from src.pages.stats import get_actuals, get_preds, find_pattern
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

    for pattern in configs.candlestick_patterns:
        (
            st.session_state[f"x_{pattern.pattern}"],
            st.session_state[f"y_{pattern.pattern}"],
        ) = find_pattern(st.session_state.df, pattern.pattern)

for pattern in configs.candlestick_patterns:
    if f"show_{pattern.pattern}" not in st.session_state:
        st.session_state[f"show_{pattern.pattern}"] = False

with col2:
    for pattern in configs.candlestick_patterns:
        if st.button(f"Find {pattern.display_name}"):
            st.session_state[f"show_{pattern.pattern}"] = not st.session_state[
                f"show_{pattern.pattern}"
            ]

with col1:
    fig = go.Figure(
        data=[st.session_state.actual_candlestick, st.session_state.pred_candlestick],
    )

    for i, pattern in enumerate(configs.candlestick_patterns):
        if st.session_state[f"show_{pattern.pattern}"]:
            color = "red" if pattern.trend == "bearish" else "green"

            fig.add_trace(
                go.Scatter(
                    x=st.session_state[f"x_{pattern.pattern}"],
                    y=(0.97 - 0.001 * i) * st.session_state[f"y_{pattern.pattern}"],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol=pattern.plotly_symbol),
                    name=f"{pattern.display_name}",
                )
            )

    st.plotly_chart(fig)
