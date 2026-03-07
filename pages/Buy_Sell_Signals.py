import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Buy / Sell Signals", layout="wide")

st.title("📈 Buy / Sell Signal Charts")

# -----------------------------
# LOAD TICKERS
# -----------------------------
@st.cache_data
def load_tickers():
    df = pd.read_csv("data/EQUITY_L.csv")
    return df["SYMBOL"].dropna().unique().tolist()

tickers = load_tickers()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Chart Settings")

ticker = st.sidebar.selectbox("Select Stock", tickers)

timeframe = st.sidebar.selectbox(
    "Time Frame",
    ["1mo","3mo","6mo","1y","2y","5y"]
)

rows = st.sidebar.slider("Number of rows to display",50,500,200)

signal_type = st.sidebar.selectbox(
    "Signal Strategy",
    ["Golden Cross","UT Bot"]
)

chart_type = st.sidebar.selectbox(
    "Chart Style",
    ["Line","Candlestick"]
)

dimension = st.sidebar.selectbox(
    "Chart Dimension",
    ["2D","3D"]
)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
symbol = ticker + ".NS"

df = yf.download(symbol, period=timeframe)

df = df.dropna()

# -----------------------------
# SIGNAL CALCULATIONS
# -----------------------------

data = df.copy()

# GOLDEN CROSS
if signal_type == "Golden Cross":

    data["20_SMA"] = data["Close"].rolling(20).mean()
    data["50_SMA"] = data["Close"].rolling(50).mean()

    data["Signal"] = 0
    data.loc[data["20_SMA"] > data["50_SMA"],"Signal"] = 1
    data["Position"] = data["Signal"].diff()

    buy = data[data["Position"] == 1]
    sell = data[data["Position"] == -1]

# UT BOT
else:

    atr_period = 10
    multiplier = 1

    data["H-L"] = data["High"] - data["Low"]
    data["H-PC"] = abs(data["High"] - data["Close"].shift())
    data["L-PC"] = abs(data["Low"] - data["Close"].shift())

    tr = data[["H-L","H-PC","L-PC"]].max(axis=1)
    atr = tr.rolling(atr_period).mean()

    data["upper"] = data["Close"] - multiplier * atr
    data["lower"] = data["Close"] + multiplier * atr

    data["trend"] = 0
    data.loc[data["Close"] > data["upper"],"trend"] = 1
    data.loc[data["Close"] < data["lower"],"trend"] = -1

    data["trend_shift"] = data["trend"].diff()

    buy = data[data["trend_shift"] == 2]
    sell = data[data["trend_shift"] == -2]

# limit rows
data = data.tail(rows)

# -----------------------------
# 2D CHART
# -----------------------------
if dimension == "2D":

    fig = go.Figure()

    if chart_type == "Candlestick":

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candles"
            )
        )

    else:

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Close Price"
            )
        )

    # BUY
    fig.add_trace(
        go.Scatter(
            x=buy.index,
            y=buy["Close"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="green"
            ),
            name="Buy"
        )
    )

    # SELL
    fig.add_trace(
        go.Scatter(
            x=sell.index,
            y=sell["Close"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="red"
            ),
            name="Sell"
        )
    )

    fig.update_layout(
        height=700,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3D CHART
# -----------------------------
else:

    data["Index"] = np.arange(len(data))

    fig = go.Figure()

    # PRICE LINE
    fig.add_trace(
        go.Scatter3d(
            x=data.index,
            y=data["Index"],
            z=data["Close"],
            mode="lines",
            name="Price"
        )
    )

    # BUY SIGNALS
    fig.add_trace(
        go.Scatter3d(
            x=buy.index,
            y=[data.loc[i,"Index"] for i in buy.index],
            z=buy["Close"],
            mode="markers",
            marker=dict(
                size=6,
                color="green",
                symbol="diamond"
            ),
            name="Buy"
        )
    )

    # SELL SIGNALS
    fig.add_trace(
        go.Scatter3d(
            x=sell.index,
            y=[data.loc[i,"Index"] for i in sell.index],
            z=sell["Close"],
            mode="markers",
            marker=dict(
                size=6,
                color="red",
                symbol="diamond"
            ),
            name="Sell"
        )
    )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis_title="Date",
            yaxis_title="Time Index",
            zaxis_title="Price"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SIGNAL TABLE
# -----------------------------

st.subheader("Recent Signals")

signal_df = pd.concat([buy, sell]).sort_index()

if len(signal_df) > 0:

    signal_df = signal_df[["Close"]].copy()
    signal_df["Signal"] = "Buy"

    signal_df.loc[sell.index,"Signal"] = "Sell"

    st.dataframe(signal_df.tail(20))

else:

    st.info("No signals generated.")
