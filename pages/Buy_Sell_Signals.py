import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Buy Sell Signals", layout="wide")

st.title("📈 Buy / Sell Signal Charts")

# ---------------------------
# LOAD TICKERS
# ---------------------------
@st.cache_data
def load_tickers():
    df = pd.read_csv("data/EQUITY_L.csv")
    return df["SYMBOL"].dropna().tolist()

tickers = load_tickers()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Chart Settings")

ticker = st.sidebar.selectbox("Stock", tickers)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1mo","3mo","6mo","1y","2y","5y"]
)

rows = st.sidebar.slider("Rows to Display", 50, 500, 200)

strategy = st.sidebar.selectbox(
    "Signal Strategy",
    ["Golden Cross","UT Bot"]
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Line","Candlestick"]
)

dimension = st.sidebar.selectbox(
    "Chart Dimension",
    ["2D","3D"]
)

z_axis = st.sidebar.selectbox(
    "3D Z Axis Variable",
    ["Close","Volatility","Returns"]
)

# ---------------------------
# DOWNLOAD DATA
# ---------------------------
symbol = ticker + ".NS"

df = yf.download(symbol, period=timeframe)

if df.empty:
    st.error("No data available")
    st.stop()

# fix multi index
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

df = df.tail(rows)

df.reset_index(inplace=True)

# ---------------------------
# EXTRA METRICS
# ---------------------------

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = df["Returns"].rolling(10).std()

# ---------------------------
# GOLDEN CROSS
# ---------------------------
if strategy == "Golden Cross":

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Signal"] = 0
    df.loc[df["SMA20"] > df["SMA50"], "Signal"] = 1

    df["Position"] = df["Signal"].diff()

    buy = df[df["Position"] == 1]
    sell = df[df["Position"] == -1]

# ---------------------------
# UT BOT
# ---------------------------
else:

    atr_period = 10
    multiplier = 1

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift())
    df["L-PC"] = abs(df["Low"] - df["Close"].shift()])

    tr = df[["H-L","H-PC","L-PC"]].max(axis=1)

    atr = tr.rolling(atr_period).mean()

    df["upper"] = df["Close"] - multiplier * atr
    df["lower"] = df["Close"] + multiplier * atr

    df["trend"] = 0
    df.loc[df["Close"] > df["upper"],"trend"] = 1
    df.loc[df["Close"] < df["lower"],"trend"] = -1

    df["trend_shift"] = df["trend"].diff()

    buy = df[df["trend_shift"] == 2]
    sell = df[df["trend_shift"] == -2]

# ---------------------------
# 2D CHART
# ---------------------------
if dimension == "2D":

    fig = go.Figure()

    if chart_type == "Candlestick":

        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles"
            )
        )

    else:

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close"
            )
        )

    # SMA lines (Golden Cross only)
    if strategy == "Golden Cross":

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA20"],
                mode="lines",
                name="SMA20"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA50"],
                mode="lines",
                name="SMA50"
            )
        )

    # BUY
    fig.add_trace(
        go.Scatter(
            x=buy["Date"],
            y=buy["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            name="Buy"
        )
    )

    # SELL
    fig.add_trace(
        go.Scatter(
            x=sell["Date"],
            y=sell["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            name="Sell"
        )
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 3D CHART
# ---------------------------
else:

    df["Index3D"] = np.arange(len(df))

    z = df[z_axis]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df["Date"],
            y=df["Close"],
            z=z,
            mode="lines",
            name="Price Path"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=buy["Date"],
            y=buy["Close"],
            z=buy[z_axis],
            mode="markers",
            marker=dict(size=6, color="green"),
            name="Buy"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=sell["Date"],
            y=sell["Close"],
            z=sell[z_axis],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Sell"
        )
    )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis_title="Date",
            yaxis_title="Price",
            zaxis_title=z_axis
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# SIGNAL TABLE
# ---------------------------

st.subheader("Recent Signals")

signals = pd.concat([buy, sell]).sort_values("Date")

if not signals.empty:

    signals = signals[["Date","Close"]].copy()

    signals["Signal"] = "Buy"
    signals.loc[sell.index,"Signal"] = "Sell"

    st.dataframe(signals.tail(20))

else:

    st.info("No signals generated.")
