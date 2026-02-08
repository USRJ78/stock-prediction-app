# Home.py
# Updated version - fixes asset resolution, yfinance reliability, mock premium debug
# FIXED: Removed st.session_state["premium_email"] = ... lines to prevent StreamlitAPIException

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from difflib import get_close_matches
from utils import advanced_ai_prediction
from datetime import date
import requests
import hashlib
import razorpay
import json
import time
from io import StringIO
from auth_store import save_premium_user
import os  # added for debug

st.set_page_config(page_title="Universal Market App", layout="wide")

# Mock database of premium users (in-memory fallback)
if "premium_users" not in st.session_state:
    st.session_state.premium_users = set()

# Razorpay setup – uses secrets
try:
    RAZORPAY_KEY_ID = st.secrets["RAZORPAY_KEY_ID"]
    RAZORPAY_KEY_SECRET = st.secrets["RAZORPAY_KEY_SECRET"]
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
except Exception as e:
    st.error("Razorpay keys not set in Streamlit secrets.")
    client = None

SUBSCRIPTION_AMOUNT = 99900  # paise → ₹999
SUBSCRIPTION_CURRENCY = "INR"
SUBSCRIPTION_NAME = "Universal Market App Premium"
SUBSCRIPTION_DESC = "AI Predictions + Advanced Features"

st.title("📊 Universal Stock & ETF Portfolio App")
st.markdown("Search by **name or ticker**, allocate capital, and run portfolio simulations.")

# Premium AI Config
API_URL = "https://universal-market-app-1.onrender.com"

def user_id_from_email(email: str) -> str:
    return hashlib.md5(email.strip().lower().encode()).hexdigest()

# Run-state fix
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

def trigger_run():
    st.session_state.run_analysis = True

# Helpers
ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/csv,*/*;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    def parse_df(df: pd.DataFrame) -> dict:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip() + ".NS"
        df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.upper().str.strip()
        return dict(zip(df["NAME OF COMPANY"], df["SYMBOL"]))

    # 1. Try live
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if not df.empty and "SYMBOL" in df.columns:
            return parse_df(df)
    except Exception:
        pass

    # 2. Local fallback
    try:
        if os.path.exists("data/EQUITY_L.csv"):
            df = pd.read_csv("data/EQUITY_L.csv")
            if not df.empty and "SYMBOL" in df.columns:
                return parse_df(df)
    except Exception:
        pass

    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_search_options():
    stock_map = load_nse_stock_list() or {}
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

@st.cache_data(ttl=300)
def load_prices(tickers, start, end, retries=2):
    tickers = sorted(list(set(tickers)))
    for attempt in range(retries + 1):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if not data.empty:
                data = data.dropna()
                data.index = pd.to_datetime(data.index)
                return data
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return pd.DataFrame()  # empty on failure

def price_scaling(raw_prices_df):
    scaled = raw_prices_df.copy()
    for col in scaled.columns:
        if scaled[col].notna().any():
            scaled[col] = scaled[col] / scaled[col].iloc[0]
    return scaled

# Sidebar
st.sidebar.header("Inputs")
search_options = load_search_options()

if len(search_options) <= len(ETF_MAP):
    st.sidebar.warning("Stock list limited. Add data/EQUITY_L.csv to repo or use manual tickers.")

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stocks / ETFs",
    options=search_options,
    key="selected_assets",
    on_change=trigger_run
)

manual_assets = st.sidebar.text_input(
    "✍️ Manual tickers / names (comma separated)",
    "",
    key="manual_assets",
    on_change=trigger_run
)

initial_amount = st.sidebar.number_input(
    "Initial Investment (INR)",
    value=100000,
    step=10000,
    key="initial_amount",
    on_change=trigger_run
)

start_date = st.sidebar.date_input(
    "Start Date",
    date(2023, 1, 1),
    key="start_date",
    on_change=trigger_run
)

end_date = st.sidebar.date_input(
    "End Date",
    date.today(),
    key="end_date",
    on_change=trigger_run
)

run_mc = st.sidebar.checkbox("Run Monte Carlo Simulation", key="run_mc", on_change=trigger_run)

num_sims = st.sidebar.number_input("No. of simulations", 1000, 20000, 5000, step=1000, key="num_sims", on_change=trigger_run)

if st.sidebar.button("Run Analysis", key="run_button"):
    st.session_state.run_analysis = True

# Premium AI
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔮 AI Prediction (Premium)")
ai_enabled = st.sidebar.checkbox("Enable AI Prediction", key="ai_enabled", on_change=trigger_run)
email = st.sidebar.text_input("Email (for premium access)", key="premium_email")

horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
horizon_label = st.sidebar.selectbox("Horizon", list(horizon_map.keys()), index=1, key="ai_horizon", on_change=trigger_run)
horizon_days = horizon_map[horizon_label]

# Legal links
st.sidebar.markdown("---")
st.sidebar.markdown("**Legal & Policies**")
st.sidebar.markdown("- [Privacy Policy](https://drive.google.com/file/d/1JLl2BzpkHDpz6-cfMR6b-wiXpcN6dKet/view?usp=sharing)")
st.sidebar.markdown("- [Terms of Service](https://drive.google.com/file/d/1VABpc6ZANgS1L3DEiSlLuPX7oqS8mh44/view?usp=sharing)")
st.sidebar.markdown("- [Refund & Cancellation Policy](https://drive.google.com/file/d/1i0g2g9YdNASv1UyweBiiNEtXyxM_7H3A/view?usp=sharing)")
st.sidebar.markdown("**Contact:** udaysinghrathore09@gmail.com")

# ────────────────────────────────────────────────
# Main logic
# ────────────────────────────────────────────────
if st.session_state.run_analysis:
    if end_date <= start_date:
        st.error("End Date must be after Start Date")
        st.stop()

    user_assets = list(selected_assets) + [x.strip() for x in manual_assets.split(",") if x.strip()]
    if not user_assets:
        st.error("Select or enter at least one asset")
        st.stop()

    # Improved resolve_assets
    stock_map = load_nse_stock_list() or {}
    resolved = {}
    for item in user_assets:
        key = str(item).upper().strip()
        if not key:
            continue
        if key.endswith(".NS"):
            resolved[item] = key
            continue
        if key in ETF_MAP:
            resolved[item] = ETF_MAP[key]
            continue
        if stock_map:
            matches = get_close_matches(key, list(stock_map.keys()), n=1, cutoff=0.65)
            if matches:
                resolved[item] = stock_map[matches[0]]
                continue
        # Last resort - assume symbol
        resolved[item] = key + ".NS"

    valid = {k: v for k, v in resolved.items() if v}
    invalid = [k for k, v in resolved.items() if not v]
    if invalid:
        st.warning(f"Could not resolve: {', '.join(invalid)}")
    if not valid:
        st.error("No valid assets resolved")
        st.stop()

    tickers = list(valid.values())
    st.subheader("Resolved Assets")
    st.write(dict(zip(valid.keys(), tickers)))

    prices = load_prices(tickers, start_date, end_date)
    if prices.empty:
        st.error("No price data fetched from Yahoo Finance. Try different dates, fewer tickers, or check your internet.")
        st.stop()

    returns = prices.pct_change().dropna()

    # Random allocation
    weights = np.random.random(len(prices.columns))
    weights /= weights.sum()
    allocation = initial_amount * weights

    alloc_df = pd.DataFrame({
        "Asset": prices.columns,
        "Weight": weights.round(4),
        "Allocation (INR)": allocation.round(2)
    })
    st.subheader("💰 Portfolio Allocation")
    st.dataframe(alloc_df)

    # Scaled prices
    scaled_prices = price_scaling(prices)
    scaled_prices["Date"] = scaled_prices.index
    st.subheader("📊 Percentage Change (Scaled)")
    fig_scaled = px.line(scaled_prices, x="Date", y=scaled_prices.columns[:-1],
                         title="Scaled Price Change (Base = 1.0)")
    st.plotly_chart(fig_scaled, use_container_width=True)

    # Actual prices
    st.subheader("📈 Price Movement (Actual)")
    raw_prices = prices.copy()
    raw_prices["Date"] = raw_prices.index
    fig_raw = px.line(raw_prices, x="Date", y=raw_prices.columns[:-1],
                      title="Actual Prices")
    st.plotly_chart(fig_raw, use_container_width=True)

    # Portfolio value over time
    portfolio_positions = (prices / prices.iloc[0]) * allocation
    portfolio_value = portfolio_positions.sum(axis=1)
    port_df = pd.DataFrame({"Date": portfolio_value.index, "Portfolio Value": portfolio_value})
    st.subheader("💼 Total Portfolio Value")
    fig_port = px.line(port_df, x="Date", y="Portfolio Value", title="Portfolio Value Over Time")
    st.plotly_chart(fig_port, use_container_width=True)

    # Monte Carlo
    if run_mc:
        st.subheader("🎯 Monte Carlo Simulation")
        mean_returns = returns.mean() * 252
        cov = returns.cov() * 252
        sim_results = []
        weight_list = []
        for _ in range(num_sims):
            w = np.random.random(len(prices.columns))
            w /= w.sum()
            weight_list.append(w)
            port_ret = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            sharpe = port_ret / port_vol if port_vol != 0 else np.nan
            sim_results.append([port_ret, port_vol, sharpe])

        sim_df = pd.DataFrame(sim_results, columns=["Return", "Volatility", "Sharpe"])
        best_idx = sim_df["Sharpe"].idxmax()
        best_return = sim_df.loc[best_idx, "Return"]
        best_vol = sim_df.loc[best_idx, "Volatility"]

        fig_mc = px.scatter(sim_df, x="Volatility", y="Return", color="Sharpe",
                            title="Monte Carlo Portfolios")
        fig_mc.add_scatter(x=[best_vol], y=[best_return], mode="markers",
                           marker=dict(size=15, color="red"), name="Best Sharpe")
        st.plotly_chart(fig_mc, use_container_width=True)

        st.subheader("Optimal Weights (Max Sharpe)")
        best_weights = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": weight_list[best_idx]
        }).sort_values("Weight", ascending=False)
        st.dataframe(best_weights)

    # ─── Premium AI Section ───
    if ai_enabled:
        st.markdown("---")
        st.subheader("🔮 AI Return Prediction (Premium)")

        if not email:
            st.warning("Enter your email in sidebar to use Premium AI.")
        else:
            chosen_ticker = tickers[0] if tickers else None
            if len(tickers) > 1:
                chosen_ticker = st.selectbox("Select asset for AI prediction", tickers)

            email_clean = email.lower().strip()
            is_premium = email_clean in st.session_state.premium_users

            if is_premium:
                st.success(f"✅ Premium Active for {email}")
                if st.button("Open AI Premium Prediction →", type="primary"):
                    st.switch_page("pages/AI_Prediction.py")
            else:
                st.info("🔒 Premium feature locked.")
                st.markdown("**Subscribe to unlock AI predictions**")

                if st.button("Subscribe for ₹999/mo via Razorpay", type="primary"):
                    if client is None:
                        st.error("Razorpay not configured.")
                    else:
                        try:
                            order = client.order.create({
                                "amount": SUBSCRIPTION_AMOUNT,
                                "currency": SUBSCRIPTION_CURRENCY,
                                "receipt": f"rcpt_{int(time.time())}",
                                "notes": {"email": email}
                            })
                            options = {
                                "key": RAZORPAY_KEY_ID,
                                "amount": SUBSCRIPTION_AMOUNT,
                                "currency": SUBSCRIPTION_CURRENCY,
                                "name": SUBSCRIPTION_NAME,
                                "description": SUBSCRIPTION_DESC,
                                "order_id": order["id"],
                                "prefill": {"email": email},
                                "theme": {"color": "#3399cc"}
                            }
                            js = f"""
                            <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
                            <script>
                                var options = {json.dumps(options)};
                                options.handler = function (response){{
                                    alert("Payment ID: " + response.razorpay_payment_id + "\\nSignature: " + response.razorpay_signature);
                                }};
                                var rzp = new Razorpay(options);
                                rzp.open();
                            </script>
                            """
                            st.components.v1.html(js, height=1)
                            st.info("Complete payment in popup → copy Payment ID & Signature below.")
                            if "pending_order" not in st.session_state:
                                st.session_state.pending_order = {}
                            st.session_state.pending_order[email_clean] = order["id"]
                        except Exception as e:
                            st.error(f"Order creation failed: {e}")

                # Verification form
                if email_clean in st.session_state.get("pending_order", {}):
                    st.markdown("### Verify Payment")
                    pid = st.text_input("Payment ID", key=f"pid_{email_clean}")
                    sig = st.text_input("Signature", key=f"sig_{email_clean}")
                    if st.button("Verify & Unlock"):
                        try:
                            params = {
                                "razorpay_order_id": st.session_state.pending_order[email_clean],
                                "razorpay_payment_id": pid,
                                "razorpay_signature": sig
                            }
                            client.utility.verify_payment_signature(params)
                            save_premium_user(email)
                            st.session_state.premium_users.add(email_clean)
                            del st.session_state.pending_order[email_clean]
                            st.success("Premium unlocked!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Verification failed: {e}")

                # Developer mock override
                with st.expander("Developer Override (Mock Payment)"):
                    if st.button(f"Simulate Successful Payment for {email}"):
                        try:
                            save_premium_user(email)
                            st.session_state.premium_users.add(email_clean)
                            st.success("Mock payment success → Premium active!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Mock failed inside save_premium_user: {str(e)}")
                            st.exception(e)  # shows traceback

else:
    st.info("Select assets / adjust inputs → graphs update automatically.")

