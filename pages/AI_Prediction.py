# pages/AI_Prediction.py
# ✅ FULL UPDATED FILE (paste & replace)
# Fix: NSE equity list loads from LIVE NSE else falls back to repo file data/EQUITY_L.csv (robust path)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from difflib import get_close_matches
from datetime import date
import requests
from io import StringIO
from pathlib import Path

# If you still use it elsewhere on this page
from utils import advanced_ai_prediction

st.set_page_config(page_title="AI Premium Prediction", layout="wide")

# ----------------------------
# Premium access gate (cross-page)
# ----------------------------
st.title("🔮 AI Premium Prediction")

# Try to reuse stored premium email from Home.py
stored_email = (st.session_state.get("premium_email") or "").strip().lower()
email = stored_email

if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai_page").strip().lower()

# premium_users may exist from Home.py session
premium_users = st.session_state.get("premium_users", set())

if (not email) or (email not in premium_users):
    st.error("Premium access required for this page. Please return to the main page and subscribe/verify.")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")
    st.stop()

st.success(f"Premium active for {email}")

# ----------------------------
# Helpers: NSE list + resolving
# ----------------------------

ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_stock_list():
    """
    Robust NSE loader for Streamlit Cloud:
    1) Try LIVE NSE CSV with headers
    2) Fallback to local repo file: ../data/EQUITY_L.csv (relative to pages/)
    Returns dict: {COMPANY_NAME_UPPER: "SYMBOL.NS"}
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }

    def parse_df(df: pd.DataFrame) -> dict:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip() + ".NS"
        df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.upper().str.strip()
        return dict(zip(df["NAME OF COMPANY"], df["SYMBOL"]))

    # 1) LIVE NSE
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if isinstance(df, pd.DataFrame) and not df.empty:
            return parse_df(df)
    except Exception:
        pass

    # 2) LOCAL fallback (AI_Prediction.py is in pages/, so go one level up)
    try:
        base_dir = Path(__file__).resolve().parent          # .../pages
        csv_path = base_dir.parent / "data" / "EQUITY_L.csv"  # .../data/EQUITY_L.csv
        df = pd.read_csv(csv_path)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return parse_df(df)
    except Exception:
        pass

    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_search_options():
    stock_map = load_nse_stock_list() or {}
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_list() or {}
    resolved = {}
    for item in user_inputs:
        key = str(item).upper().strip()
        if not key:
            continue
        if "." in key:
            resolved[item] = key
        elif key in ETF_MAP:
            resolved[item] = ETF_MAP[key]
        else:
            matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6) if stock_map else []
            resolved[item] = stock_map[matches[0]] if matches else None
    return resolved

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(tickers, start, end):
    tickers = sorted(list(set(tickers)))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

# ----------------------------
# UI: Value Picks + Prediction panel
# ----------------------------

st.markdown("## ✅ Value Picks (NSE)")

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    max_stocks = st.number_input("How many stocks to shortlist?", min_value=10, max_value=200, value=40, step=10)

with colB:
    start_date = st.date_input("Start date (for correlation)", value=date(2021, 1, 1))

with colC:
    end_date = st.date_input("End date (for correlation)", value=date.today())

st.caption("Tip: If you updated the CSV today, go to Streamlit Cloud → Manage app → Clear cache.")

# Load tickers list (never crashes)
stock_map = load_nse_stock_list() or {}

if not stock_map:
    st.error(
        "Could not load NSE equity list.\n\n"
        "✅ Make sure the file exists in your repo root at: `data/EQUITY_L.csv`\n"
        "✅ Then Streamlit Cloud → Manage app → Clear cache → Restart"
    )
    st.stop()

# Basic selection UI (kept because you sometimes still want it)
search_options = load_search_options()
selected_assets = st.multiselect(
    "Optional: Pick specific stocks/ETFs (or leave blank to use automatic shortlist)",
    options=search_options,
    default=[]
)

manual_assets = st.text_input(
    "Optional: Manual tickers (comma separated, e.g., RELIANCE.NS, TCS.NS)",
    ""
)

user_assets = list(selected_assets) + [x.strip() for x in manual_assets.split(",") if x.strip()]
resolved = resolve_assets(user_assets) if user_assets else {}
valid_tickers_manual = [v for v in resolved.values() if v] if resolved else []

# ----------------------------
# AUTOMATIC SHORTLIST (simple + stable)
# ----------------------------
# Since you didn't paste your full valuation formulas here, this shortlist is a robust placeholder:
# - Uses top "max_stocks" by Market Cap from yfinance info (best-effort)
# - You can later swap this shortlist with your Peter Lynch + Graham logic cleanly.

@st.cache_data(ttl=3600, show_spinner=False)
def shortlist_large_liquid_stocks(stock_map: dict, n: int) -> list:
    # We sample from stock_map values and try to rank by marketCap quickly.
    # yfinance info may be missing for some tickers; we skip those.
    tickers = list(stock_map.values())
    # Don't explode requests; take a manageable slice first
    tickers = tickers[: min(len(tickers), 600)]

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap", None)
            if mc is not None:
                rows.append((t, mc))
        except Exception:
            continue

        # stop early once we have enough candidates to sort
        if len(rows) >= max(n * 4, 200):
            break

    rows.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in rows[:n]]

auto_run = st.button("Run Value Picks Screening", type="primary")

if "value_picks_ran" not in st.session_state:
    st.session_state.value_picks_ran = False
if "value_picks_tickers" not in st.session_state:
    st.session_state.value_picks_tickers = []

if auto_run:
    with st.spinner("Building shortlist..."):
        tickers_auto = shortlist_large_liquid_stocks(stock_map, int(max_stocks))
        st.session_state.value_picks_tickers = tickers_auto
        st.session_state.value_picks_ran = True

# If user manually selected tickers, prefer those; else use the stored shortlist after run
tickers_to_use = valid_tickers_manual if valid_tickers_manual else st.session_state.value_picks_tickers

if st.session_state.value_picks_ran and tickers_to_use:
    st.success(f"Using {len(tickers_to_use)} tickers for correlation & portfolio simulation.")
    st.write(", ".join(tickers_to_use[:40]) + (" ..." if len(tickers_to_use) > 40 else ""))

    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    # ----------------------------
    # Correlation Heatmap (like Home.py)
    # ----------------------------
    prices = load_prices(tickers_to_use, start_date, end_date)

    if prices.empty or prices.shape[1] < 2:
        st.error("Not enough price data for correlation (try fewer stocks or a different date range).")
        st.stop()

    returns = prices.pct_change().dropna()

    st.markdown("### 🔥 Correlation Heatmap")
    corr = returns.corr()

    fig_heat = px.imshow(
        corr,
        aspect="auto",
        title="Correlation Heatmap (Daily Returns)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ----------------------------
    # Pick least correlated subset (simple greedy)
    # ----------------------------
    st.markdown("### 🧠 Least-Correlated Basket")
    basket_size = st.slider("Basket size", min_value=3, max_value=min(20, prices.shape[1]), value=min(10, prices.shape[1]))

    def pick_least_correlated(corr_df: pd.DataFrame, k: int) -> list:
        cols = list(corr_df.columns)
        # start from the stock with lowest average correlation
        avg_corr = corr_df.abs().mean().sort_values()
        selected = [avg_corr.index[0]]
        remaining = [c for c in cols if c not in selected]

        while len(selected) < k and remaining:
            # pick candidate that minimizes avg corr with selected
            best = None
            best_score = None
            for c in remaining:
                score = corr_df.loc[c, selected].abs().mean()
                if best_score is None or score < best_score:
                    best_score = score
                    best = c
            selected.append(best)
            remaining.remove(best)
        return selected

    basket = pick_least_correlated(corr, int(basket_size))
    st.write("Selected basket:", ", ".join(basket))

    # ----------------------------
    # Portfolio Monte Carlo on selected basket (like Home.py)
    # ----------------------------
    st.markdown("### 🎯 Monte Carlo Simulation (Basket)")
    initial_amount = st.number_input("Initial Investment (INR)", value=100000, step=10000)
    num_sims = st.number_input("No. of simulations", min_value=500, max_value=20000, value=5000, step=500)

    basket_prices = prices[basket]
    basket_returns = basket_prices.pct_change().dropna()

    mean_returns = basket_returns.mean() * 252
    cov = basket_returns.cov() * 252

    sim_results = []
    weight_list = []

    with st.spinner("Running simulations..."):
        for _ in range(int(num_sims)):
            w = np.random.random(len(basket))
            w /= w.sum()
            weight_list.append(w)

            port_return = float(np.dot(w, mean_returns))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            sharpe = (port_return / port_vol) if port_vol != 0 else np.nan
            sim_results.append([port_return, port_vol, sharpe])

    sim_out_df = pd.DataFrame(sim_results, columns=["Portfolio_Return", "Volatility", "Sharpe_Ratio"])
    sharpe_series = sim_out_df["Sharpe_Ratio"].replace([np.inf, -np.inf], np.nan)
    optimal_idx = sharpe_series.idxmax()

    optimal_portfolio_return = float(sim_out_df.loc[optimal_idx, "Portfolio_Return"])
    optimal_volatility = float(sim_out_df.loc[optimal_idx, "Volatility"])

    fig_mc = px.scatter(
        sim_out_df,
        x="Volatility",
        y="Portfolio_Return",
        color="Sharpe_Ratio",
        size="Sharpe_Ratio",
        hover_data=["Sharpe_Ratio"],
        title="Monte Carlo: Return vs Volatility (Sharpe colored)"
    )
    fig_mc.add_trace(go.Scatter(
        x=[optimal_volatility],
        y=[optimal_portfolio_return],
        mode="markers",
        name="Optimal Point",
        marker=dict(size=30, color="red")
    ))
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown("#### ✅ Optimal Portfolio Weights (Max Sharpe)")
    best_df = pd.DataFrame({
        "Asset": basket,
        "Weight": weight_list[int(optimal_idx)],
        "Allocation (INR)": (np.array(weight_list[int(optimal_idx)]) * float(initial_amount)).round(2)
    })
    st.dataframe(best_df)

    # ----------------------------
    # Optional: AI prediction (single ticker)
    # ----------------------------
    st.markdown("---")
    st.markdown("## 🔮 AI Return Prediction (Single Asset)")

    horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()), index=1)
    horizon_days = horizon_map[horizon_label]

    chosen_ticker = st.selectbox("Select asset for prediction", basket, index=0)

    if st.button("Run AI Prediction"):
        with st.spinner(f"AI Agent is analyzing {chosen_ticker}..."):
            try:
                ai_df, analysis = advanced_ai_prediction(chosen_ticker, days=horizon_days)

                current_data = yf.Ticker(chosen_ticker).history(period="1d")
                if not current_data.empty:
                    current_price = float(current_data["Close"].iloc[-1])

                    last_pred = float(ai_df["Predicted_Price"].iloc[-1])
                    last_lower = float(ai_df["Lower_Bound"].iloc[-1])
                    last_upper = float(ai_df["Upper_Bound"].iloc[-1])

                    ret = (last_pred - current_price) / current_price
                    ret_low = (last_lower - current_price) / current_price
                    ret_high = (last_upper - current_price) / current_price

                    st.metric("Predicted Return", f"{ret*100:.2f}%")
                    st.write(
                        f"Confidence Range: {ret_low*100:.2f}% to {ret_high*100:.2f}% "
                        f"(Horizon: {horizon_days} trading days)"
                    )

                    st.markdown("### AI Analysis")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend", analysis.get("Trend", "—"))
                    c2.metric("Volatility", analysis.get("Volatility", "—"))
                    c3.metric("Confidence", analysis.get("Confidence_Score", "—"))
                    st.info(f"Recommendation: **{analysis.get('Recommendation','—')}**")

                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Predicted_Price"],
                        name="AI Prediction"
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Upper_Bound"],
                        fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Lower_Bound"],
                        fill="tonexty", mode="lines", line_color="rgba(0,0,0,0)",
                        name="Confidence Interval"
                    ))
                    st.plotly_chart(fig_ai, use_container_width=True)

                else:
                    st.error("Could not fetch current price for return calculation.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

else:
    st.info("Click **Run Value Picks Screening** to build picks, correlation heatmap, and portfolio simulation.")

# ----------------------------
# Navigation
# ----------------------------
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")
