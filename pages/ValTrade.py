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
    1, 20, 5
)

stop_loss_pct = st.sidebar.slider(
    "Stop Loss %",
    0.01, 0.50, 0.10, 0.01
)

lookback_period = st.sidebar.selectbox(
    "Lookback Period",
    ["1y", "2y", "5y", "10y"],
    index=3
)

valuation_buffer = st.sidebar.slider(
    "Valuation Buffer",
    1.0, 2.0, 1.25, 0.05
)

market_cap_filter = st.sidebar.selectbox(
    "Market Cap Filter",
    ["All", "Large Cap", "Mid Cap", "Small Cap"]
)

# ---------- LOAD STOCKS ----------
@st.cache_data
def load_nse_tickers():
    df = pd.read_csv("data/EQUITY_L.csv")
    symbols = df["SYMBOL"].dropna().astype(str).str.strip().unique()
    return [s + ".NS" for s in symbols if "&" not in s]

tickers = load_nse_tickers()

# ---------- UT BOT ----------
def compute_utbot(df, atr_period=10, multiplier=2):
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


# ---------- MARKET CAP FILTER ----------
def passes_market_cap_filter(market_cap):
    if market_cap is None:
        return False

    market_cap_cr = market_cap / 10000000  # convert to crores

    if market_cap_filter == "All":
        return True
    elif market_cap_filter == "Large Cap":
        return market_cap_cr > 20000
    elif market_cap_filter == "Mid Cap":
        return 5000 <= market_cap_cr <= 20000
    elif market_cap_filter == "Small Cap":
        return market_cap_cr < 5000

    return False


# ---------- RUN ----------
if st.button("Run Backtest"):

    cash = initial_capital
    open_positions = {}
    trade_log = []
    all_data = {}

    progress = st.progress(0)

    # ---------- LOAD DATA ----------
    for idx, ticker in enumerate(tickers):

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            market_cap = info.get("marketCap")

            if not passes_market_cap_filter(market_cap):
                continue

            eps = info.get("trailingEps")
            if eps is None:
                continue

            growth = info.get("earningsQuarterlyGrowth", 0.05)
            g = min(growth * 100, 12)

            intrinsic = eps * (8.5 + 2 * g)

            df = stock.history(period=lookback_period)

            if df.empty:
                continue

            df = compute_utbot(df)
            df["Intrinsic"] = intrinsic

            all_data[ticker] = df

        except:
            continue

        progress.progress((idx + 1) / len(tickers))

    all_dates = sorted(set(d for df in all_data.values() for d in df.index))
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
        slots = max_positions - len(open_positions)

        if slots > 0:
            candidates = []

            for ticker, df in all_data.items():
                if ticker in open_positions:
                    continue
                if current_date not in df.index:
                    continue

                row = df.loc[current_date]

                if row["buy"] and row["Close"] < row["Intrinsic"] * valuation_buffer:
                    discount = (row["Intrinsic"] - row["Close"]) / row["Intrinsic"]
                    candidates.append((ticker, discount))

            candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, _ in candidates[:slots]:
                row = all_data[ticker].loc[current_date]
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
            if current_date in all_data[ticker].index:
                portfolio_value += pos["shares"] * all_data[ticker].loc[current_date]["Close"]

        equity_curve.append(portfolio_value)

    # ---------- RESULTS ----------
    trades = pd.DataFrame(trade_log)

    if not trades.empty:
        final_capital = equity_curve[-1]
        win_rate = (trades["Profit"] > 0).mean()

        st.subheader("Results")
        st.write("Initial Capital:", initial_capital)
        st.write("Final Capital:", round(final_capital, 2))
        st.write("Return %:", round((final_capital / initial_capital - 1) * 100, 2))
        st.write("Win Rate:", round(win_rate * 100, 2), "%")

        fig, ax = plt.subplots()
        ax.plot(equity_curve)
        ax.set_title("Equity Curve")
        st.pyplot(fig)

        st.subheader("Trade Log")
        st.dataframe(trades)

    else:
        st.warning("No trades generated.")
