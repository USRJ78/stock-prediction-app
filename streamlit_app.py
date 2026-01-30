# (Updated code with Razorpay integration)

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
import hashlib
import razorpay
import json
import time

st.set_page_config(page_title="Universal Market App", layout="wide")

# Mock database of premium users (in-memory → resets on redeploy/sleep)
if "premium_users" not in st.session_state:
    st.session_state.premium_users = set()

# ─── Razorpay Configuration ───────────────────────────────────────
# Add these to Streamlit Cloud → Settings → Secrets management
# Format:
# RAZORPAY_KEY_ID = "rzp_test_xxxxxxxxxxxxxx"
# RAZORPAY_KEY_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxx"
try:
    RAZORPAY_KEY_ID = st.secrets["RAZORPAY_KEY_ID"]
    RAZORPAY_KEY_SECRET = st.secrets["RAZORPAY_KEY_SECRET"]
    RAZORPAY_CLIENT = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
except Exception:
    st.error("Razorpay secrets not configured. Please add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in Streamlit Cloud secrets.")
    RAZORPAY_CLIENT = None

SUBSCRIPTION_AMOUNT   = 99900       # paise → ₹999 (change as needed)
SUBSCRIPTION_CURRENCY = "INR"
SUBSCRIPTION_NAME     = "Universal Market App Premium"
SUBSCRIPTION_DESC     = "AI Predictions + Advanced Features"

# Check for payment success simulation (optional dev tool)
query_params = st.query_params
if "dev_premium" in query_params and query_params["dev_premium"] == "true":
    if "premium_email" in st.session_state:
        st.session_state.premium_users.add(st.session_state.premium_email)
        st.success("Dev mode: Premium activated!")

st.title("📊 Universal Stock & ETF Portfolio App")
st.markdown("Search by **name or ticker**, allocate capital, and run portfolio simulations.")

# ============================
# ✅ Premium AI Config
# ============================
API_URL = "https://universal-market-app-1.onrender.com"  # change if needed

def user_id_from_email(email: str) -> str:
    return hashlib.md5(email.strip().lower().encode()).hexdigest()

# Run-state fix
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

def trigger_run():
    st.session_state.run_analysis = True

# Helpers (unchanged) ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df["SYMBOL"] = df["SYMBOL"].astype(str) + ".NS"
    return dict(zip(df["NAME OF COMPANY"].str.upper(), df["SYMBOL"]))

ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_list()
    resolved = {}
    for item in user_inputs:
        key = item.upper().strip()
        if "." in key:
            resolved[item] = key
        elif key in ETF_MAP:
            resolved[item] = ETF_MAP[key]
        else:
            matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6)
            resolved[item] = stock_map[matches[0]] if matches else None
    return resolved

@st.cache_data(ttl=300)
def load_prices(tickers, start, end):
    tickers = sorted(list(set(tickers)))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

def plot_financial_data(df, title):
    fig = px.line(title=title)
    for col in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[col], name=col)
    fig.update_traces(line_width=3)
    fig.update_layout({'plot_bgcolor': "white"})
    st.plotly_chart(fig, use_container_width=True)

def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[1:]:
        scaled_prices_df[i] = raw_prices_df[i] / raw_prices_df[i].iloc[0]
    return scaled_prices_df

# Sidebar (unchanged except premium part) ────────────────────────────
st.sidebar.header("Inputs")

@st.cache_data(ttl=3600)
def load_search_options():
    stock_map = load_nse_stock_list()
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

search_options = load_search_options()

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stocks / ETFs (recommended)",
    options=search_options,
    key="selected_assets",
    on_change=trigger_run
)

manual_assets = st.sidebar.text_input(
    "✍️ Or manually type names / tickers (comma separated)",
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
    date(2021, 1, 1),
    key="start_date",
    on_change=trigger_run
)

end_date = st.sidebar.date_input(
    "End Date",
    date.today(),
    key="end_date",
    on_change=trigger_run
)

run_mc = st.sidebar.checkbox(
    "Run Monte Carlo Simulation",
    key="run_mc",
    on_change=trigger_run
)

num_sims = st.sidebar.number_input(
    "No. of simulations",
    1000,
    20000,
    5000,
    step=1000,
    key="num_sims",
    on_change=trigger_run
)

if st.sidebar.button("Run Analysis", key="run_button"):
    st.session_state.run_analysis = True

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔮 AI Prediction (Premium)")

ai_enabled = st.sidebar.checkbox("Enable AI Prediction", key="ai_enabled", on_change=trigger_run)

email = st.sidebar.text_input("Email (for premium access)", key="premium_email")

horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
horizon_label = st.sidebar.selectbox("Horizon", list(horizon_map.keys()), index=1, key="ai_horizon", on_change=trigger_run)
horizon_days = horizon_map[horizon_label]

# ─── Main App ───────────────────────────────────────────────────────
if st.session_state.run_analysis:
    # ... (your existing analysis code remains completely unchanged)
    # I'll skip repeating the whole analysis block for brevity.
    # Keep everything from:
    #     if end_date <= start_date:   →   up to the end of Monte Carlo block

    # ============================
    # 🔮 Premium AI Prediction Panel (Main)
    # ============================
    if ai_enabled:
        st.markdown("---")
        st.subheader("🔮 AI Return Prediction (Premium)")

        if not email:
            st.warning("Enter your email in the sidebar to use the Premium AI feature.")
        else:
            chosen_ticker = tickers[0]
            if len(tickers) > 1:
                chosen_ticker = st.selectbox("Select asset for prediction", tickers, index=0)

            is_premium = email in st.session_state.premium_users

            if not is_premium:
                st.info("🔒 This feature is locked.")
                st.markdown(f"**Upgrade to Premium** to unlock AI predictions for {chosen_ticker}.")
                st.markdown("""
                - Advanced Volatility Analysis  
                - Confidence Intervals  
                - AI Recommendations
                """)

                # ─── Razorpay Subscribe Button ──────────────────────────────
                if st.button("Subscribe for ₹999/mo via Razorpay", type="primary"):
                    if RAZORPAY_CLIENT is None:
                        st.error("Razorpay not configured.")
                    else:
                        try:
                            order_data = {
                                "amount": SUBSCRIPTION_AMOUNT,
                                "currency": SUBSCRIPTION_CURRENCY,
                                "receipt": f"rcpt_{user_id_from_email(email)}_{int(time.time())}",
                                "notes": {"email": email, "purpose": "premium_unlock"}
                            }
                            order = RAZORPAY_CLIENT.order.create(data=order_data)
                            order_id = order["id"]

                            checkout_options = {
                                "key": RAZORPAY_KEY_ID,
                                "amount": SUBSCRIPTION_AMOUNT,
                                "currency": SUBSCRIPTION_CURRENCY,
                                "name": SUBSCRIPTION_NAME,
                                "description": SUBSCRIPTION_DESC,
                                "order_id": order_id,
                                "prefill": {
                                    "name": email.split("@")[0].title() if "@" in email else "User",
                                    "email": email,
                                },
                                "theme": {"color": "#3399cc"}
                            }

                            # JavaScript to open checkout
                            checkout_script = f"""
                            <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
                            <script>
                                var options = {json.dumps(checkout_options)};
                                options.handler = function (response) {{
                                    alert("Payment successful!\\nPayment ID: " + response.razorpay_payment_id);
                                    window.parent.postMessage({{
                                        type: "razorpay_success",
                                        order_id: "{order_id}",
                                        payment_id: response.razorpay_payment_id,
                                        signature: response.razorpay_signature
                                    }}, "*");
                                }};
                                var rzp = new Razorpay(options);
                                rzp.open();
                            </script>
                            """

                            st.components.v1.html(checkout_script, height=1)

                            st.info("""
                            Payment window opened.  
                            Complete payment → then return here and use the verification section below.
                            """)

                            # Store order_id for verification
                            st.session_state["current_order_id"] = order_id
                            st.session_state["current_email"] = email

                        except Exception as e:
                            st.error(f"Could not create Razorpay order: {str(e)}")

                # ─── Manual Verification (reliable on Streamlit Cloud) ──────
                if "current_order_id" in st.session_state:
                    st.markdown("### Verify Payment (after completing payment)")
                    payment_id = st.text_input("Razorpay Payment ID", "")
                    signature  = st.text_input("Razorpay Signature", "")

                    if st.button("Verify Payment"):
                        try:
                            params = {
                                "razorpay_order_id": st.session_state["current_order_id"],
                                "razorpay_payment_id": payment_id,
                                "razorpay_signature": signature
                            }
                            RAZORPAY_CLIENT.utility.verify_payment_signature(params)
                            # Success!
                            st.session_state.premium_users.add(st.session_state["current_email"])
                            st.success(f"Payment verified! Premium unlocked for {st.session_state['current_email']} 🎉")
                            # Clean up
                            for key in ["current_order_id", "current_email"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                        except razorpay.errors.SignatureVerificationError:
                            st.error("Invalid signature — payment verification failed.")
                        except Exception as e:
                            st.error(f"Verification error: {str(e)}")

            else:
                st.success(f"✅ Premium Active for {email}")

                if st.button("Run AI Prediction"):
                    with st.spinner(f"AI Agent is analyzing {chosen_ticker}..."):
                        try:
                            ai_df, analysis = advanced_ai_prediction(chosen_ticker, days=horizon_days)
                            current_data = yf.Ticker(chosen_ticker).history(period="1d")
                            
                            if not current_data.empty:
                                current_price = current_data['Close'].iloc[-1]
                                last_pred  = ai_df['Predicted_Price'].iloc[-1]
                                last_lower = ai_df['Lower_Bound'].iloc[-1]
                                last_upper = ai_df['Upper_Bound'].iloc[-1]
                                
                                ret      = (last_pred - current_price) / current_price
                                ret_low  = (last_lower - current_price) / current_price
                                ret_high = (last_upper - current_price) / current_price
                                
                                st.metric("Predicted Return", f"{ret*100:.2f}%")
                                st.write(f"Confidence Range: {ret_low*100:.2f}% to {ret_high*100:.2f}% (Horizon: {horizon_days} trading days)")
                                
                                st.markdown("### AI Analysis")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Trend", analysis["Trend"])
                                c2.metric("Volatility", analysis["Volatility"])
                                c3.metric("Confidence", analysis["Confidence_Score"])
                                
                                st.info(f"Recommendation: **{analysis['Recommendation']}**")
                                
                                fig_ai = go.Figure()
                                fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Predicted_Price'], name='AI Prediction', line=dict(color='purple')))
                                fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Upper_Bound'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                                fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Lower_Bound'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval', fillcolor='rgba(128, 0, 128, 0.2)'))
                                st.plotly_chart(fig_ai, use_container_width=True)
                                
                            else:
                                st.error("Could not fetch current price.")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

else:
    st.info("👈 Select assets / change dates — graphs will auto-update. (You can also click Run Analysis.)")
