import streamlit as st
from src.configs import input_form_page, analyze_page

pg = st.navigation([input_form_page, analyze_page])
pg.run()
