import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Backtest", layout="wide")

st.title("📈 Diversified UT Bot + Valuation Backtest")

# ---------- SIDEBAR ----------
st.sidebar.header("Backtest Settings")

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=10000,
    max_value=100000000,
    value=100000,
    step=10000
)

max_positions = st.sidebar.slider(
    "Maximum Open Positions",
    min_value=1,
    max_value=20,
    value=5
)

stop_loss_pct = st.sidebar.slider(
    "Stop Loss %",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01
)

lookback_period = st.sidebar.selectbox(
    "Lookback Period",
    ["1y", "2y", "5y", "10y"],
    index=3
)

stock_limit = st.sidebar.slider(
    "Number of Stocks to Scan",
    min_value=10,
    max_value=500,
    value=100
)

# ---------- LOAD NSE STOCKS ----------
@st.cache_data
def load_nse_tickers():
    df = pd.read_csv("data/EQUITY_L.csv")

    symbols = (
        df["SYMBOL"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    tickers = []

    for symbol in symbols:
        if "&" in symbol:
            continue
        tickers.append(symbol + ".NS")

    return tickers

tickers = load_nse_tickers()[:stock_limit]

# ---------- UT BOT ----------
def compute_utbot(df, atr_period=1, multiplier=1):

    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )

    df["atr"] = df["tr"].rolling(atr_period).mean()

    df["upper"] = df["Close"] - multiplier * df["atr"]
    df["lower"] = df["Close"] + multiplier * df["atr"]

    trend = [1]

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["lower"].iloc[i-1]:
            trend.append(1)
        elif df["Close"].iloc[i] < df["upper"].iloc[i-1]:
            trend.append(-1)
        else:
            trend.append(trend[-1])

    df["trend"] = trend
    df["buy"] = (df["trend"] == 1) & (df["trend"].shift() == -1)
    df["sell"] = (df["trend"] == -1) & (df["trend"].shift() == 1)

    return df

# ---------- RUN BUTTON ----------
if st.button("Run Backtest"):

    cash = initial_capital
    trade_log = []
    open_positions = {}
    all_data = {}

    progress = st.progress(0)

    # ---------- LOAD DATA ----------
    for idx, ticker in enumerate(tickers):

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=lookback_period)

            if df.empty:
                continue

            df = compute_utbot(df)

            info = stock.info
            eps = info.get("trailingEps")

            if eps is None:
                continue

            growth = info.get("earningsQuarterlyGrowth")
            if growth is None:
                growth = 0.05

            g = min(growth * 100, 12)

            intrinsic = eps * (8.5 + 2 * g)

            df["Intrinsic"] = intrinsic

            all_data[ticker] = df

        except:
            continue

        progress.progress((idx + 1) / len(tickers))

    # ---------- MASTER DATES ----------
    all_dates = sorted(
        set(date for df in all_data.values() for date in df.index)
    )

    equity_curve = []

    # ---------- BACKTEST ----------
    for current_date in all_dates:

        # SELL
        for ticker in list(open_positions.keys()):

            df = all_data[ticker]

            if current_date not in df.index:
                continue

            row = df.loc[current_date]
            pos = open_positions[ticker]

            stop_price = pos["entry_price"] * (1 - stop_loss_pct)

            if row["Close"] <= stop_price:
                exit_reason = "Stop Loss"

            elif row["sell"]:
                exit_reason = "UT Sell"

            else:
                continue

            exit_price = row["Close"]
            proceeds = pos["shares"] * exit_price
            profit = proceeds - pos["invested"]

            cash += proceeds

            trade_log.append({
                "Stock": ticker,
                "Entry Date": pos["entry_date"],
                "Exit Date": current_date,
                "Entry": pos["entry_price"],
                "Exit": exit_price,
                "Invested": pos["invested"],
                "Profit": profit,
                "Return %": (exit_price / pos["entry_price"] - 1) * 100,
                "Exit Reason": exit_reason
            })

            del open_positions[ticker]

        # BUY
        available_slots = max_positions - len(open_positions)

        if available_slots > 0:

            candidates = []

            for ticker, df in all_data.items():

                if ticker in open_positions:
                    continue

                if current_date not in df.index:
                    continue

                row = df.loc[current_date]

                if row["buy"] and row["Close"] < row["Intrinsic"]:
                    discount = (row["Intrinsic"] - row["Close"]) / row["Intrinsic"]
                    candidates.append((ticker, discount))

            candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, _ in candidates[:available_slots]:

                df = all_data[ticker]
                row = df.loc[current_date]

                allocation = cash / (max_positions - len(open_positions))

                if allocation <= 0:
                    break

                shares = allocation / row["Close"]

                cash -= allocation

                open_positions[ticker] = {
                    "entry_date": current_date,
                    "entry_price": row["Close"],
                    "shares": shares,
                    "invested": allocation
                }

        # EQUITY
        portfolio_value = cash

        for ticker, pos in open_positions.items():
            df = all_data[ticker]

            if current_date in df.index:
                portfolio_value += pos["shares"] * df.loc[current_date]["Close"]

        equity_curve.append(portfolio_value)

    # ---------- RESULTS ----------
    trades = pd.DataFrame(trade_log)

    if not trades.empty:

        trades = trades.sort_values("Exit Date")
        trades["Cumulative Profit"] = trades["Profit"].cumsum()
        trades["Equity"] = initial_capital + trades["Cumulative Profit"]

        final_capital = equity_curve[-1]
        win_rate = len(trades[trades["Profit"] > 0]) / len(trades)

        st.subheader("📊 Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Initial Capital", f"₹{initial_capital:,.0f}")
        col2.metric("Final Capital", f"₹{final_capital:,.0f}")
        col3.metric("Return %", f"{((final_capital/initial_capital)-1)*100:.2f}%")
        col4.metric("Win Rate", f"{win_rate*100:.2f}%")

        # ---------- CHART ----------
        fig, ax = plt.subplots()
        ax.plot(equity_curve)
        ax.set_title("Portfolio Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Portfolio Value")
        st.pyplot(fig)

        # ---------- TRADE TABLE ----------
        st.subheader("📋 Trade Log")
        st.dataframe(trades, use_container_width=True)

        # ---------- DOWNLOAD ----------
        csv = trades.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Trade Log CSV",
            csv,
            "trade_log.csv",
            "text/csv"
        )

    else:
        st.warning("No trades generated.")
