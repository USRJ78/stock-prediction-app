# pages/3D_Charts.py
# 3D Stock "Money Cloud" Visualizer (Time-Price-Pressure)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="3D Stock Visualizer", layout="wide")
st.title("🧭 3D Stock Visualizer — Time × Price × Pressure")

# -------------------------------------------------
# Load NSE Equity Master from data folder
# -------------------------------------------------
@st.cache_data
def load_equity_master():
    base_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_path, "data", "EQUITY_L.csv")

    if not os.path.exists(file_path):
        st.error(f"EQUITY_L.csv not found at: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["SYMBOL"])
    df["Ticker"] = df["SYMBOL"].astype(str) + ".NS"
    return df

equity_df = load_equity_master()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    ticker = st.selectbox(
        "Select Ticker",
        options=sorted(equity_df["Ticker"].unique())
    )

    period = st.selectbox(
        "Lookback Period",
        ["6mo", "1y", "2y", "5y"],
        index=1
    )

    y_mode = st.selectbox(
        "Y-Axis",
        ["Close Price", "Log Close"],
        index=0
    )

    pressure_mode = st.selectbox(
        "Z-Axis (Pressure Variable)",
        [
            "Rolling Volatility (21D, Annualized %)",
            "Volume Z-Score (20D)",
            "RSI (14)",
            "Drawdown (252D high)",
        ],
        index=0
    )

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x.squeeze()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900)
def load_ohlcv(ticker, period):
    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

# -------------------------------------------------
# Load Stock Data
# -------------------------------------------------
df = load_ohlcv(ticker, period)

if df.empty:
    st.error("No data found for selected ticker.")
    st.stop()

close = to_series(df["Close"])
volume = to_series(df["Volume"])
returns = close.pct_change() * 100

# Y Axis
if y_mode == "Log Close":
    y = np.log(close)
    y_label = "Log(Close)"
else:
    y = close
    y_label = "Close"

# Z Axis
if pressure_mode.startswith("Rolling Volatility"):
    z = close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    z_label = "Volatility %"

elif pressure_mode.startswith("Volume Z-Score"):
    z = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    z_label = "Volume Z"

elif pressure_mode.startswith("RSI"):
    z = rsi(close)
    z_label = "RSI"

else:
    rolling_max = close.rolling(252).max()
    z = (close / rolling_max - 1) * 100
    z_label = "Drawdown %"

# Ensure everything is 1D
plot_df = pd.DataFrame({
    "Date": df.index,
    "Y": to_series(y),
    "Z": to_series(z),
    "Ret%": to_series(returns)
}).dropna()

plot_df["t"] = np.arange(len(plot_df))

# -------------------------------------------------
# 3D Plot
# -------------------------------------------------
fig = go.Figure()

customdata = np.column_stack((
    plot_df["Date"].dt.strftime("%Y-%m-%d"),
    plot_df["Y"],
    plot_df["Z"],
    plot_df["Ret%"]
))

hover_template = (
    "Date: %{customdata[0]}<br>"
    + y_label + ": %{customdata[1]:.4f}<br>"
    + z_label + ": %{customdata[2]:.4f}<br>"
    + "Return: %{customdata[3]:.2f}%<br>"
    + "<extra></extra>"
)

fig.add_trace(go.Scatter3d(
    x=plot_df["t"],
    y=plot_df["Y"],
    z=plot_df["Z"],
    mode="lines+markers",
    marker=dict(size=3),
    hovertemplate=hover_template,
    customdata=customdata
))

fig.update_layout(
    height=750,
    title=f"{ticker} — 3D Time × {y_label} × {z_label}",
    scene=dict(
        xaxis_title="Time Index",
        yaxis_title=y_label,
        zaxis_title=z_label
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Data Table
# -------------------------------------------------
st.markdown("## 📊 View Data Table")

row_option = st.selectbox(
    "Select number of rows to display",
    ["Last 10", "Last 25", "Last 50", "Last 100", "All Rows"],
    index=0
)

rows_map = {
    "Last 10": 10,
    "Last 25": 25,
    "Last 50": 50,
    "Last 100": 100,
    "All Rows": None
}

rows_to_show = rows_map[row_option]

with st.expander("Show Data"):
    if rows_to_show:
        st.dataframe(plot_df.tail(rows_to_show), use_container_width=True)
    else:
        st.dataframe(plot_df, use_container_width=True)
