from datetime import datetime, timedelta
import os
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION ET STYLE
# ============================================================
st.set_page_config(
    page_title="Smart Home Energy Orchestration",
    page_icon="SH",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F8F6F1;
        --navy: #0F1E36;
        --blue: #1D3557;
        --muted: #64748B;
        --border: #E5E0D8;
        --card: #FFFFFF;
        --green: #286348;
        --red: #A83D35;
        --sand: #F1ECE3;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--navy);
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 0.9rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #263754;
    }

    section[data-testid="stSidebar"] * {
        color: #F8F6F1 !important;
    }

    section[data-testid="stSidebar"] input {
        color: var(--navy) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F8F6F1 !important;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 10px 13px;
        box-shadow: 0 2px 7px rgba(15, 30, 54, 0.035);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.035em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-size: 1.35rem !important;
        font-weight: 750 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
