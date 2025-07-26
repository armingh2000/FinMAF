import streamlit as st

analyze_page = st.Page("pages/analyze.py")
input_form_page = st.Page("pages/input_form.py")

from collections import namedtuple

# Define the named tuple structure
CandlestickPattern = namedtuple(
    "CandlestickPattern", ["pattern", "trend", "display_name", "plotly_symbol"]
)

# Original list of tuples
candlestick_patterns_classified_with_symbols = [
    ("bearish_engulfing", "bearish", "Bearish Engulfing", "triangle-down"),
    ("bearish_harami", "bearish", "Bearish Harami", "triangle-down"),
    ("bullish_engulfing", "bullish", "Bullish Engulfing", "triangle-up"),
    ("bullish_harami", "bullish", "Bullish Harami", "triangle-up"),
    ("dark_cloud_cover", "bearish", "Dark Cloud Cover", "x"),
    ("doji", "bullish", "Doji", "diamond"),
    ("doji_star", "bullish", "Doji Star", "diamond"),
    ("dragonfly_doji", "bullish", "Dragonfly Doji", "diamond-open"),
    ("gravestone_doji", "bearish", "Gravestone Doji", "diamond-open"),
    ("hammer", "bullish", "Hammer", "circle"),
    ("hanging_man", "bearish", "Hanging Man", "x"),
    ("inverted_hammer", "bullish", "Inverted Hammer", "circle-open"),
    ("morning_star", "bullish", "Morning Star", "star"),
    ("morning_star_doji", "bullish", "Morning Star Doji", "star-open"),
    ("piercing_pattern", "bullish", "Piercing Pattern", "triangle-up"),
    ("rain_drop", "bullish", "Rain Drop", "circle"),
    ("rain_drop_doji", "bullish", "Rain Drop Doji", "circle-open"),
    ("shooting_star", "bearish", "Shooting Star", "triangle-down-open"),
    ("star", "bullish", "Star", "star"),
]

# Convert the list of tuples to a list of named tuples
candlestick_patterns = [
    CandlestickPattern(*pattern)
    for pattern in candlestick_patterns_classified_with_symbols
]
