# pages/3D_Charts.py
# ✅ FIXED: handles yfinance MultiIndex / DataFrame columns (Close becomes Series)
# 3D Stock "Money Cloud" Visualizer (Time-Price-Pressure)

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
def _to_series(x) -> pd.Series:
    """Convert x to a 1D Series safely (handles DataFrame single column)."""
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # If DataFrame has one column, squeeze it to Series
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # If multiple columns, try common patterns (Close / Adj Close)
        for col in ["Close", "Adj Close"]:
            if col in x.columns:
                s = x[col]
                if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                    return s.iloc[:, 0]
                if isinstance(s, pd.Series):
                    return s
        # fallback: first column
        return x.iloc[:, 0]
    # numpy / list-like
    return pd.Series(x)

def safe_series(x) -> pd.Series:
    s = _to_series(x)
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = safe_series(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out

def zscore(x: pd.Series, window: int = 20) -> pd.Series:
    x = safe_series(x)
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd

@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False, group_by="column")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)

    # If MultiIndex columns appear, flatten them:
    # Example: ('Close', 'RELIANCE.NS') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer first level names like Open/High/Low/Close/Volume
        df.columns = [c[0] for c in df.columns]

    return df

def pick_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Get a column as Series robustly."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if name in df.columns:
        return safe_series(df[name])
    # Try case variations
    for c in df.columns:
        if str(c).strip().lower() == name.lower():
            return safe_series(df[c])
    return pd.Series(dtype=float)

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

close = pick_col(df, "Close")
volume = pick_col(df, "Volume")

if close.empty:
    st.error("Could not read Close prices from Yahoo Finance response for this ticker.")
    st.stop()

rets = safe_series(close.pct_change() * 100)

# Y axis choice
if y_mode == "Log Close":
    y = safe_series(np.log(close))
    y_label = "Log(Close)"
else:
    y = safe_series(close)
    y_label = "Close"

# Compute pressure Z
if pressure_mode.startswith("Rolling Volatility"):
    vol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    z = safe_series(vol)
    z_label = "Volatility % (ann.)"

elif pressure_mode.startswith("Volume Z-Score"):
    if volume.empty:
        st.warning("Volume not available for this ticker. Switching Z-axis to Rolling Volatility.")
        vol = close.pct_change().rolling(21).std() * np.sqrt(252) * 100
        z = safe_series(vol)
        z_label = "Volatility % (ann.)"
    else:
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
    st.warning("Not enough data to compute selected indicators. Try a longer period (e.g., 2y or 5y).")
    st.stop()

plot_df = plot_df.reset_index().rename(columns={"index": "Date"})
plot_df["t"] = np.arange(len(plot_df))  # numeric time index

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

customdata = np.stack([
    plot_df["Date"].dt.strftime("%Y-%m-%d"),
    plot_df["Y"].to_numpy(),
    plot_df["Z"].to_numpy(),
    plot_df["Ret%"].to_numpy(),
], axis=1)

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
        customdata=customdata
    ))

if show_points:
    fig.add_trace(go.Scatter3d(
        x=plot_df["t"],
        y=plot_df["Y"],
        z=plot_df["Z"],
        mode="markers",
        name="Points",
        marker=dict(
            size=point_size,
            color=c,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=c_label)
        ),
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            f"{y_label}: %{customdata[1]:.4f}<br>"
            f"{z_label}: %{customdata[2]:.4f}<br>"
            "Ret%: %{customdata[3]:.2f}%<br>"
            "<extra></extra>"
        ),
        customdata=customdata
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
# Interpretation hint
# -----------------------------
st.markdown("## 🧠 Interpreting 'Low Pressure Zones'")
if pressure_mode.startswith("Rolling Volatility"):
    st.write("Low pressure = **volatility compression** (quiet regime). Expansion often follows.")
elif pressure_mode.startswith("Volume Z-Score"):
    st.write("Low pressure = **low participation**. Watch for volume spike + price break.")
elif pressure_mode.startswith("RSI"):
    st.write("Low pressure = **weak momentum**. Pressure builds when RSI rises from low values.")
else:
    st.write("Low pressure = **deep drawdown** (pain). Reversion often starts when drawdown stabilizes.")

with st.expander("Show computed data (last 10 rows)"):
    st.dataframe(plot_df.tail(10), use_container_width=True)
