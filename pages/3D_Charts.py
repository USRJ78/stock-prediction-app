# pages/3D_Charts.py
# 3D Stock "Money Cloud" Visualizer (Time-Price-Pressure)
# X = Time index
# Y = Price (or log price)
# Z = "Pressure" variable (volatility / volume shock / RSI / drawdown)
#
# Works with Indian tickers like RELIANCE.NS

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Stock Visualizer", layout="wide")
st.title("🧭 3D Stock Visualizer — Time × Price × Pressure")

# -----------------------------
# Helpers
# -----------------------------
def safe_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out

def zscore(x: pd.Series, window: int = 20) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd

@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", "RELIANCE.NS").strip().upper()
    period = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=1)

    y_mode = st.selectbox("Y-Axis", ["Close Price", "Log Close"], index=0)

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

# -----------------------------
# Load data
# -----------------------------
df = load_ohlcv(ticker, period)

if df.empty:
    st.error("No data returned. Check ticker (e.g., RELIANCE.NS) or try a different period.")
    st.stop()

# Basic series
close = safe_series(df["Close"])
volume = safe_series(df.get("Volume", pd.Series(index=df.index, dtype=float)))

rets = close.pct_change() * 100
rets = safe_series(rets)

# Y axis choice
if y_mode == "Log Close":
    y = np.log(close)
    y_label = "Log(Close)"
else:
    y = close
    y_label = "Close"

y = safe_series(pd.Series(y, index=df.index))

# Compute pressure Z
if pressure_mode.startswith("Rolling Volatility"):
    # annualized volatility from daily returns (close-to-close)
    vol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    z = safe_series(vol)
    z_label = "Volatility % (ann.)"

elif pressure_mode.startswith("Volume Z-Score"):
    z = safe_series(zscore(volume, 20))
    z_label = "Volume Z"

elif pressure_mode.startswith("RSI"):
    z = safe_series(rsi(close, 14))
    z_label = "RSI"

else:  # Drawdown
    roll_max = close.rolling(252).max()
    dd = (close / roll_max - 1.0) * 100
    z = safe_series(dd)
    z_label = "Drawdown %"

# Combine and drop NaNs from indicators
plot_df = pd.DataFrame({
    "Close": close,
    "Y": y,
    "Z": z,
    "Ret%": rets,
    "Volume": volume
}, index=df.index).dropna()

if plot_df.empty:
    st.warning("Not enough data to compute selected indicators. Try a longer period (e.g., 2y).")
    st.stop()

# X axis: numeric time index (Plotly 3D prefers numbers)
plot_df = plot_df.reset_index().rename(columns={"index": "Date"})
plot_df["t"] = np.arange(len(plot_df))  # 0..N-1

# Color scale source
if color_mode == "Daily Return %":
    c = plot_df["Ret%"]
    c_label = "Daily Return %"
else:
    c = plot_df["Z"]
    c_label = z_label

# -----------------------------
# 3D Plot
# -----------------------------
fig = go.Figure()

if show_line:
    fig.add_trace(go.Scatter3d(
        x=plot_df["t"],
        y=plot_df["Y"],
        z=plot_df["Z"],
        mode="lines",
        name="Path",
        line=dict(width=line_width),
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            f"{y_label}: %{customdata[1]:.4f}<br>"
            f"{z_label}: %{customdata[2]:.4f}<br>"
            "Ret%: %{customdata[3]:.2f}%<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([
            plot_df["Date"].dt.strftime("%Y-%m-%d"),
            plot_df["Y"],
            plot_df["Z"],
            plot_df["Ret%"],
        ], axis=1)
    ))

if show_points:
    fig.add_trace(go.Scatter3d(
        x=plot_df["t"],
        y=plot_df["Y"],
        z=plot_df["Z"],
        mode="markers",
        name="Points",
        marker=dict(size=point_size, color=c, colorscale="Viridis", showscale=True, colorbar=dict(title=c_label)),
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            f"{y_label}: %{customdata[1]:.4f}<br>"
            f"{z_label}: %{customdata[2]:.4f}<br>"
            "Ret%: %{customdata[3]:.2f}%<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([
            plot_df["Date"].dt.strftime("%Y-%m-%d"),
            plot_df["Y"],
            plot_df["Z"],
            plot_df["Ret%"],
        ], axis=1)
    ))

fig.update_layout(
    height=750,
    title=f"{ticker} — 3D Time × {y_label} × {z_label}",
    scene=dict(
        xaxis_title="Time Index (old → new)",
        yaxis_title=y_label,
        zaxis_title=z_label
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# “Low pressure zone” hint box (tradable interpretation)
# -----------------------------
st.markdown("## 🧠 Interpreting 'Low Pressure Zones'")
if pressure_mode.startswith("Rolling Volatility"):
    st.write(
        "Low pressure = **volatility compression**. "
        "When Z (vol) is low and price starts moving up, it often signals a breakout regime."
    )
elif pressure_mode.startswith("Volume Z-Score"):
    st.write(
        "Low pressure = **low attention / low participation**. "
        "When volume is unusually low (negative Z-score) and later spikes, it can mark accumulation → expansion."
    )
elif pressure_mode.startswith("RSI"):
    st.write(
        "Low pressure = **weak momentum** (low RSI). "
        "Momentum rising from low RSI can act like pressure building and releasing."
    )
else:
    st.write(
        "Low pressure = **deep drawdown** (pain). "
        "Mean reversion often starts when drawdown stabilizes and volatility falls."
    )

# Small data preview
with st.expander("Show computed data (last 10 rows)"):
    st.dataframe(plot_df.tail(10), use_container_width=True)
