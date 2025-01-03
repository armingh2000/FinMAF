import streamlit as st
from src.configs import analyze_page
from src.data.download import download_symbols

symbols, _ = download_symbols()

option = st.selectbox(
    "Please select the symbol to analyze",
    symbols,
    index=None,
    placeholder="Select symbol ...",
)


submit_button = st.button("Analyze")

if submit_button and option:
    st.session_state["symbol"] = option
    st.switch_page(analyze_page)
