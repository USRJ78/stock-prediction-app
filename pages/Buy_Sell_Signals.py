# pages/Buy_Sell_Signals.py
# Golden Cross Buy/Sell Signal Visualizer

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Buy/Sell Signals", layout="wide")
st.title("📈 Golden Cross Buy/Sell Signal Dashboard")

# -------------------------------------------------
# Load NSE Equity Master
# -------------------------------------------------
@st.cache_data
def load_equity_master():
    base_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_path, "data", "EQUITY_L.csv")

    if not os.path.exists(file_path):
        st.error("EQUITY_L.csv not found.")
        st.stop()

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["SYMBOL"])
    df["Ticker"] = df["SYMBOL"].astype(str) + ".NS"
    return df

equity_df = load_equity_master()

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:

    st.header("Strategy Settings")

    ticker = st.selectbox(
        "Select Ticker",
        sorted(equity_df["Ticker"].unique())
    )

    period = st.selectbox(
        "Historical Period",
        ["6mo", "1y", "2y", "5y"],
        index=1
    )

    short_window = st.slider(
        "Short SMA Window",
        5, 50, 20
    )

    long_window = st.slider(
        "Long SMA Window",
        20, 200, 50
    )

    data_points = st.slider(
        "Number of Recent Rows",
        50, 500, 200
    )

# -------------------------------------------------
# Load Stock Data
# -------------------------------------------------
@st.cache_data(ttl=900)
def load_stock(ticker, period):

    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

df = load_stock(ticker, period)

if df.empty:
    st.error("No stock data available.")
    st.stop()

# -------------------------------------------------
# Prepare Data
# -------------------------------------------------
close = df["Close"]

df["SMA_short"] = close.rolling(short_window).mean()
df["SMA_long"] = close.rolling(long_window).mean()

df["Signal"] = 0
df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = 1

df["Position"] = df["Signal"].diff()

# Filter recent data
rdt = df.tail(data_points)

buy = rdt[rdt["Position"] == 1]
sell = rdt[rdt["Position"] == -1]

# -------------------------------------------------
# Plot Chart
# -------------------------------------------------
fig = go.Figure()

# Close price
fig.add_trace(go.Scatter(
    x=rdt.index,
    y=rdt["Close"],
    name="Close Price",
    line=dict(color="black")
))

# Short SMA
fig.add_trace(go.Scatter(
    x=rdt.index,
    y=rdt["SMA_short"],
    name=f"{short_window}-Day SMA",
    line=dict(color="blue")
))

# Long SMA
fig.add_trace(go.Scatter(
    x=rdt.index,
    y=rdt["SMA_long"],
    name=f"{long_window}-Day SMA",
    line=dict(color="green")
))

# Buy signals
fig.add_trace(go.Scatter(
    x=buy.index,
    y=buy["SMA_short"],
    mode="markers",
    name="Buy Signal",
    marker=dict(
        symbol="triangle-up",
        size=14,
        color="green"
    )
))

# Sell signals
fig.add_trace(go.Scatter(
    x=sell.index,
    y=sell["SMA_short"],
    mode="markers",
    name="Sell Signal",
    marker=dict(
        symbol="triangle-down",
        size=14,
        color="red"
    )
))

fig.update_layout(
    height=700,
    title=f"{ticker} — Golden Cross Strategy",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Signal Table
# -------------------------------------------------
st.markdown("## 📊 Signal Log")

signal_df = rdt[(rdt["Position"] == 1) | (rdt["Position"] == -1)].copy()

signal_df["Signal Type"] = signal_df["Position"].apply(
    lambda x: "BUY" if x == 1 else "SELL"
)

signal_df = signal_df[["Close", "Signal Type"]]

st.dataframe(signal_df)

# -------------------------------------------------
# Strategy Explanation
# -------------------------------------------------
st.markdown("## 🧠 Strategy Logic")

st.write("""
This strategy uses a **Golden Cross system**:

- **Buy Signal** → When the short SMA crosses **above** the long SMA
- **Sell Signal** → When the short SMA crosses **below** the long SMA

Typical parameters:
- Short SMA = 20
- Long SMA = 50
""")
