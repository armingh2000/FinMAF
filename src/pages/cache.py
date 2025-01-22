import torch
from src.model.model import FinMAF
import streamlit as st
import src.configs as configs


@st.cache_resource
def get_model():
    model = FinMAF(configs.data_dim)
    print("Loading model weights...")
    model.load_state_dict(torch.load(configs.model_path))

    return model
