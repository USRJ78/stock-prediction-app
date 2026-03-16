import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Buy Sell Signals", layout="wide")

st.title("📈 Buy / Sell Signal Charts")

# -------------------------
# LOAD TICKERS
# -------------------------
@st.cache_data
def load_tickers():
    df = pd.read_csv("data/EQUITY_L.csv")
    return df["SYMBOL"].dropna().tolist()

tickers = load_tickers()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Chart Settings")

ticker = st.sidebar.selectbox("Stock", tickers)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1mo","3mo","6mo","1y","2y","5y","4h"]
)

rows = st.sidebar.slider("Rows to Display",50,500,200)

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
    "3D Axis Variable",
    ["Close","Volatility","Returns","Volume","Momentum"]
)

# -------------------------
# DOWNLOAD DATA
# -------------------------
symbol = ticker + ".NS"

if timeframe == "4h":
    df = yf.download(symbol, interval="4h", period="60d")
else:
    df = yf.download(symbol, period=timeframe)

if df.empty:
    st.error("No data found")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()
df = df.tail(rows)
df.reset_index(inplace=True)

# -------------------------
# EXTRA METRICS
# -------------------------
df["Returns"] = df["Close"].pct_change()
df["Volatility"] = df["Returns"].rolling(10).std()
df["Momentum"] = df["Close"] - df["Close"].shift(10)

# -------------------------
# SIGNAL STRATEGIES
# -------------------------
if strategy == "Golden Cross":

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    df["Signal"] = 0
    df.loc[df["SMA20"] > df["SMA50"], "Signal"] = 1

    df["Position"] = df["Signal"].diff()

    buy = df[df["Position"] == 1]
    sell = df[df["Position"] == -1]

else:

    atr_period = 1
    key_value = 2

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift())
    df["L-PC"] = abs(df["Low"] - df["Close"].shift())

    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(atr_period).mean()

    nLoss = key_value * df["ATR"]

    trailing_stop = [df["Close"].iloc[0]]

    for i in range(1,len(df)):

        prev_stop = trailing_stop[i-1]
        close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i-1]

        if close > prev_stop and prev_close > prev_stop:
            stop = max(prev_stop, close - nLoss.iloc[i])

        elif close < prev_stop and prev_close < prev_stop:
            stop = min(prev_stop, close + nLoss.iloc[i])

        elif close > prev_stop:
            stop = close - nLoss.iloc[i]

        else:
            stop = close + nLoss.iloc[i]

        trailing_stop.append(stop)

    df["UT_Stop"] = trailing_stop

    df["Signal"] = 0

    df.loc[
        (df["Close"] > df["UT_Stop"]) &
        (df["Close"].shift() <= df["UT_Stop"].shift()),
        "Signal"
    ] = 1

    df.loc[
        (df["Close"] < df["UT_Stop"]) &
        (df["Close"].shift() >= df["UT_Stop"].shift()),
        "Signal"
    ] = -1

    buy = df[df["Signal"] == 1]
    sell = df[df["Signal"] == -1]

# -------------------------
# 2D CHART
# -------------------------
if dimension == "2D":

    fig = go.Figure()

    if chart_type == "Candlestick":

        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ))

    else:

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price"
        ))

    if strategy == "Golden Cross":

        fig.add_trace(go.Scatter(x=df["Date"],y=df["SMA20"],mode="lines",name="SMA20"))
        fig.add_trace(go.Scatter(x=df["Date"],y=df["SMA50"],mode="lines",name="SMA50"))

    if strategy == "UT Bot":

        fig.add_trace(go.Scatter(x=df["Date"],y=df["UT_Stop"],mode="lines",name="UT Stop"))

    fig.add_trace(go.Scatter(
        x=buy["Date"],
        y=buy["Close"],
        mode="markers",
        marker=dict(symbol="triangle-up",size=12,color="green"),
        name="Buy"
    ))

    fig.add_trace(go.Scatter(
        x=sell["Date"],
        y=sell["Close"],
        mode="markers",
        marker=dict(symbol="triangle-down",size=12,color="red"),
        name="Sell"
    ))

    fig.update_layout(height=700,xaxis_title="Date",yaxis_title="Price")

    st.plotly_chart(fig,use_container_width=True)

# -------------------------
# BACKTEST SECTION
# -------------------------
st.sidebar.header("Backtest Settings")

investment = st.sidebar.number_input("Investment Amount ₹",1000,10000000,100000)

start_date = st.sidebar.date_input("Start Date",df["Date"].min())

end_date = st.sidebar.date_input("End Date",df["Date"].max())

bt_df = df[(df["Date"] >= pd.to_datetime(start_date)) &
           (df["Date"] <= pd.to_datetime(end_date))].copy()

cash = investment
shares = 0

equity_curve = []
trades = []

for i in range(len(bt_df)):

    row = bt_df.iloc[i]

    if row["Signal"] == 1 and cash > 0:

        shares = cash / row["Close"]
        cash = 0

        trades.append({
            "Date":row["Date"],
            "Type":"BUY",
            "Price":row["Close"]
        })

    elif row["Signal"] == -1 and shares > 0:

        cash = shares * row["Close"]
        shares = 0

        trades.append({
            "Date":row["Date"],
            "Type":"SELL",
            "Price":row["Close"],
            "Value":cash
        })

    portfolio_value = cash if shares == 0 else shares * row["Close"]
    equity_curve.append(portfolio_value)

bt_df["Equity"] = equity_curve

final_value = equity_curve[-1]

profit = final_value - investment

return_pct = (profit/investment)*100

# -------------------------
# MAX DRAWDOWN
# -------------------------
rolling_max = bt_df["Equity"].cummax()
drawdown = bt_df["Equity"] / rolling_max - 1
max_drawdown = drawdown.min() * 100

# -------------------------
# WIN RATE
# -------------------------
wins = 0
losses = 0

for i in range(1,len(trades)):

    if trades[i]["Type"] == "SELL" and trades[i-1]["Type"] == "BUY":

        buy_price = trades[i-1]["Price"]
        sell_price = trades[i]["Price"]

        if sell_price > buy_price:
            wins += 1
        else:
            losses += 1

total_trades = wins + losses
win_rate = (wins/total_trades)*100 if total_trades > 0 else 0

# -------------------------
# PERFORMANCE METRICS
# -------------------------
st.subheader("Strategy Performance")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Initial Investment",f"₹{investment:,.0f}")
col2.metric("Final Value",f"₹{final_value:,.0f}")
col3.metric("Return %",f"{return_pct:.2f}%")
col4.metric("Max Drawdown",f"{max_drawdown:.2f}%")

st.metric("Win Rate",f"{win_rate:.2f}%")

# -------------------------
# EQUITY CURVE
# -------------------------
st.subheader("Portfolio Growth")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=bt_df["Date"],
    y=bt_df["Equity"],
    mode="lines",
    name="Portfolio Value"
))

fig2.update_layout(height=500,xaxis_title="Date",yaxis_title="Portfolio Value")

st.plotly_chart(fig2,use_container_width=True)

# -------------------------
# TRADE HISTORY
# -------------------------
st.subheader("Trade History")

trades_df = pd.DataFrame(trades)

if not trades_df.empty:
    st.dataframe(trades_df)
else:
    st.info("No trades executed.")

# -------------------------
# SIGNAL TABLE
# -------------------------
st.subheader("Recent Signals")

signals = pd.concat([buy,sell]).sort_values("Date")

if not signals.empty:

    signals = signals[["Date","Close"]].copy()

    signals["Signal"] = "Buy"
    signals.loc[sell.index,"Signal"] = "Sell"

    st.dataframe(signals.tail(20))

else:
    st.info("No signals generated.")
