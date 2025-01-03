import streamlit as st

import os
import sys

# Get the current directory
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from src.configs import input_form_page, analyze_page

pg = st.navigation([input_form_page, analyze_page])
pg.run()
