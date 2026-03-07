# pages/Buy_Sell_Signals.py
# Strategy Signals Dashboard (Golden Cross + UT Bot)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Strategy Signals", layout="wide")
st.title("📈 Strategy Signals Dashboard")

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

    st.header("Market Settings")

    ticker = st.selectbox(
        "Select Ticker",
        sorted(equity_df["Ticker"].unique())
    )

    period = st.selectbox(
        "Historical Period",
        ["1mo","3mo","6mo","1y","2y","5y"],
        index=3
    )

    interval = st.selectbox(
        "Timeframe",
        ["1d","1h","30m","15m","5m"],
        index=0
    )

    chart_type = st.selectbox(
        "Chart Type",
        ["Candlestick","Line"]
    )

    st.divider()

    st.header("Strategy")

    strategy = st.selectbox(
        "Choose Strategy",
        ["Golden Cross","UT Bot Alerts"]
    )

    st.divider()

    st.header("Golden Cross Settings")

    short_window = st.slider(
        "Short SMA",
        5, 50, 20
    )

    long_window = st.slider(
        "Long SMA",
        20, 200, 50
    )

    st.divider()

    st.header("UT Bot Settings")

    atr_period = st.slider(
        "ATR Period",
        5, 30, 10
    )

    multiplier = st.slider(
        "ATR Multiplier",
        0.5, 5.0, 1.0
    )

    data_points = st.slider(
        "Recent Data Points",
        100, 1000, 300
    )


# -------------------------------------------------
# Load Stock Data
# -------------------------------------------------
@st.cache_data(ttl=900)
def load_stock(ticker, period, interval):

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


df = load_stock(ticker, period, interval)

if df.empty:
    st.error("No data available.")
    st.stop()


# -------------------------------------------------
# Indicators
# -------------------------------------------------
close = df["Close"]

# ---- Golden Cross
df["SMA_short"] = close.rolling(short_window).mean()
df["SMA_long"] = close.rolling(long_window).mean()

df["Signal"] = 0
df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = 1
df["Position"] = df["Signal"].diff()


# ---- ATR (for UT Bot)
high = df["High"]
low = df["Low"]

df["H-L"] = high - low
df["H-PC"] = abs(high - close.shift())
df["L-PC"] = abs(low - close.shift())

tr = df[["H-L","H-PC","L-PC"]].max(axis=1)
atr = tr.rolling(atr_period).mean()

# UT trailing stop
df["UT_Stop"] = close - multiplier * atr

df["UT_Signal"] = 0
df.loc[close > df["UT_Stop"], "UT_Signal"] = 1

df["UT_Position"] = df["UT_Signal"].diff()

# -------------------------------------------------
# Filter recent rows
# -------------------------------------------------
rdt = df.tail(data_points)

# Golden cross signals
buy_gc = rdt[rdt["Position"] == 1]
sell_gc = rdt[rdt["Position"] == -1]

# UT signals
buy_ut = rdt[rdt["UT_Position"] == 1]
sell_ut = rdt[rdt["UT_Position"] == -1]

# -------------------------------------------------
# Plot Chart
# -------------------------------------------------
fig = go.Figure()

# Chart type
if chart_type == "Candlestick":

    fig.add_trace(go.Candlestick(
        x=rdt.index,
        open=rdt["Open"],
        high=rdt["High"],
        low=rdt["Low"],
        close=rdt["Close"],
        name="Price"
    ))

else:

    fig.add_trace(go.Scatter(
        x=rdt.index,
        y=rdt["Close"],
        name="Close",
        line=dict(color="black")
    ))


# -------------------------------------------------
# Strategy Plots
# -------------------------------------------------
if strategy == "Golden Cross":

    fig.add_trace(go.Scatter(
        x=rdt.index,
        y=rdt["SMA_short"],
        name=f"{short_window} SMA",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=rdt.index,
        y=rdt["SMA_long"],
        name=f"{long_window} SMA",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=buy_gc.index,
        y=buy_gc["SMA_short"],
        mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", size=14, color="green")
    ))

    fig.add_trace(go.Scatter(
        x=sell_gc.index,
        y=sell_gc["SMA_short"],
        mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", size=14, color="red")
    ))

    signal_df = rdt[(rdt["Position"] == 1) | (rdt["Position"] == -1)].copy()
    signal_df["Signal"] = signal_df["Position"].apply(lambda x: "BUY" if x==1 else "SELL")


# -------------------------------------------------
# UT BOT
# -------------------------------------------------
else:

    fig.add_trace(go.Scatter(
        x=rdt.index,
        y=rdt["UT_Stop"],
        name="UT Trailing Stop",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=buy_ut.index,
        y=buy_ut["Close"],
        mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", size=14, color="green")
    ))

    fig.add_trace(go.Scatter(
        x=sell_ut.index,
        y=sell_ut["Close"],
        mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", size=14, color="red")
    ))

    signal_df = rdt[(rdt["UT_Position"] == 1) | (rdt["UT_Position"] == -1)].copy()
    signal_df["Signal"] = signal_df["UT_Position"].apply(lambda x: "BUY" if x==1 else "SELL")


fig.update_layout(
    height=750,
    title=f"{ticker} — {strategy}",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Signal Table
# -------------------------------------------------
st.markdown("## 📊 Signal Log")

signal_df = signal_df[["Close","Signal"]]

st.dataframe(signal_df)

# -------------------------------------------------
# Strategy Explanation
# -------------------------------------------------
st.markdown("## 🧠 Strategy Info")

if strategy == "Golden Cross":

    st.write("""
Golden Cross strategy:

• BUY → Short SMA crosses ABOVE Long SMA  
• SELL → Short SMA crosses BELOW Long SMA
""")

else:

    st.write("""
UT Bot Alerts logic:

• Uses ATR-based trailing stop  
• BUY → Price crosses above trailing stop  
• SELL → Price crosses below trailing stop
""")
