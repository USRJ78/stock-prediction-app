# pages/3D_Charts.py
# 3D Stock "Money Cloud" Visualizer (Time-Price-Pressure)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Stock Visualizer", layout="wide")
st.title("🧭 3D Stock Visualizer — Time × Price × Pressure")

# -------------------------------------------------
# Load NSE Equity Master (Same as Home Page)
# -------------------------------------------------
@st.cache_data
def load_equity_master():
    df = pd.read_csv("EQUITY_L.csv")
    df = df.dropna(subset=["SYMBOL"])
    df["Ticker"] = df["SYMBOL"].astype(str) + ".NS"
    return df

equity_df = load_equity_master()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    # Ticker Dropdown (Searchable)
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

    st.markdown("---")

    line_width = st.slider("3D Line Width", 1, 10, 4)
    point_size = st.slider("Point Size", 1, 8, 3)

    show_points = st.checkbox("Show points", value=True)
    show_line = st.checkbox("Show line path", value=True)

    color_mode = st.selectbox(
        "Color points by",
        ["Daily Return %", "Pressure (Z)"],
        index=0
    )

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------------------------
# Load Stock Data
# -------------------------------------------------
@st.cache_data(ttl=900)
def load_ohlcv(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    return df

df = load_ohlcv(ticker, period)

if df.empty:
    st.error("No data found for selected ticker.")
    st.stop()

close = df["Close"]
volume = df["Volume"]
returns = close.pct_change() * 100

# Y Axis
if y_mode == "Log Close":
    y = np.log(close)
    y_label = "Log(Close)"
else:
    y = close
    y_label = "Close"

# Z Axis (Pressure)
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

plot_df = pd.DataFrame({
    "Date": df.index,
    "Y": y,
    "Z": z,
    "Ret%": returns
}).dropna()

plot_df["t"] = np.arange(len(plot_df))

# Color Selection
if color_mode == "Daily Return %":
    color_data = plot_df["Ret%"]
    color_label = "Daily Return %"
else:
    color_data = plot_df["Z"]
    color_label = z_label

# -------------------------------------------------
# 3D Plot
# -------------------------------------------------
fig = go.Figure()

customdata = np.stack([
    plot_df["Date"].dt.strftime("%Y-%m-%d"),
    plot_df["Y"],
    plot_df["Z"],
    plot_df["Ret%"],
], axis=1)

hover_template = (
    "Date: %{customdata[0]}<br>"
    + y_label + ": %{customdata[1]:.4f}<br>"
    + z_label + ": %{customdata[2]:.4f}<br>"
    + "Return: %{customdata[3]:.2f}%<br>"
    + "<extra></extra>"
)

if show_line:
    fig.add_trace(go.Scatter3d(
        x=plot_df["t"],
        y=plot_df["Y"],
        z=plot_df["Z"],
        mode="lines",
        line=dict(width=line_width),
        hovertemplate=hover_template,
        customdata=customdata,
        name="Path"
    ))

if show_points:
    fig.add_trace(go.Scatter3d(
        x=plot_df["t"],
        y=plot_df["Y"],
        z=plot_df["Z"],
        mode="markers",
        marker=dict(
            size=point_size,
            color=color_data,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=color_label)
        ),
        hovertemplate=hover_template,
        customdata=customdata,
        name="Points"
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
# Data Table Section
# -------------------------------------------------
st.markdown("## 📊 View Data Table")

row_option = st.selectbox(
    "Select number of rows to display",
    ["Last 10", "Last 25", "Last 50", "Last 100", "All Rows"],
    index=0
)

if row_option == "Last 10":
    rows_to_show = 10
elif row_option == "Last 25":
    rows_to_show = 25
elif row_option == "Last 50":
    rows_to_show = 50
elif row_option == "Last 100":
    rows_to_show = 100
else:
    rows_to_show = None

with st.expander("Show Data"):
    if rows_to_show:
        st.dataframe(plot_df.tail(rows_to_show), use_container_width=True)
    else:
        st.dataframe(plot_df, use_container_width=True)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------
st.markdown("## 🧠 Interpreting Low Pressure Zones")

if pressure_mode.startswith("Rolling Volatility"):
    st.write("Low volatility zones often precede expansion.")
elif pressure_mode.startswith("Volume Z-Score"):
    st.write("Low volume zones indicate weak participation.")
elif pressure_mode.startswith("RSI"):
    st.write("Low RSI suggests oversold conditions.")
else:
    st.write("Large drawdowns may present mean reversion opportunities.")
