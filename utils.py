# utils.py
# Shared utilities - NSE stock list loader used by both Home.py and AI_Prediction.py

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_stock_list():
    """
    Load NSE equity master list.
    Priority: 1) Live URL  2) Local data/EQUITY_L.csv
    Returns: dict { "NAME OF COMPANY UPPER": "SYMBOL.NS" } or empty dict on failure
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/csv,*/*;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    def parse_df(df: pd.DataFrame) -> dict:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip() + ".NS"
        df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.upper().str.strip()
        return dict(zip(df["NAME OF COMPANY"], df["SYMBOL"]))

    # 1. Live attempt
    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if not df.empty and "SYMBOL" in df.columns:
            return parse_df(df)
    except Exception:
        pass

    # 2. Local fallback
    local_path = "data/EQUITY_L.csv"
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            if not df.empty and "SYMBOL" in df.columns:
                return parse_df(df)
    except Exception:
        pass

    return {}
