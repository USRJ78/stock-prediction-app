import streamlit as st
import yfinance as yf
from utils import advanced_ai_prediction
from difflib import get_close_matches

# ─── Reuse the same helpers from home page ─────────────────────────────────
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
def load_search_options():
    stock_map = load_nse_stock_list()
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

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

# ─── Page content ──────────────────────────────────────────────────────────

st.title("🔮 AI Premium Prediction")

# Premium check with fallback
email = st.session_state.get("premium_email", None)
if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai")

if not email or email not in st.session_state.premium_users:
    st.error("You need premium access to use this page. Please go back to the main page and subscribe/verify.")
    if st.button("← Back to Main Portfolio"):
        st.switch_page("Home.py")  # ← change if your main file name is different
    st.stop()

st.success(f"Premium active for {email}")

# ─── Stock / ETF selector – same as home page ──────────────────────────────
st.markdown("### Select Stock(s) / ETF(s)")

search_options = load_search_options()

selected_assets = st.multiselect(
    "🔍 Search & select stocks / ETFs",
    options=search_options,
    default=[],  # can prefill later if you want
    key="ai_page_selected_assets"
)

manual_assets = st.text_input(
    "✍️ Or manually type names / tickers (comma separated)",
    "",
    key="ai_page_manual_assets"
)

# Combine selections
user_assets = list(selected_assets) + [x.strip() for x in manual_assets.split(",") if x.strip()]

# Resolve to tickers
if user_assets:
    resolved = resolve_assets(user_assets)
    valid_tickers = [v for v in resolved.values() if v]
    
    if valid_tickers:
        # Show resolved tickers
        st.write("Resolved tickers:", valid_tickers)
        
        # Let user choose which one to predict (default = first)
        chosen_ticker = st.selectbox(
            "Predict for which asset?",
            options=valid_tickers,
            index=0,
            format_func=lambda x: x  # can make nicer later
        )
    else:
        st.warning("No valid tickers resolved from your selection.")
        chosen_ticker = None
else:
    chosen_ticker = st.text_input("Enter a single ticker (fallback)", "RELIANCE.NS").upper()

# ─── Horizon selection ─────────────────────────────────────────────────────
horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()), index=1)
horizon_days = horizon_map[horizon_label]

# ─── Run prediction ────────────────────────────────────────────────────────
if st.button("Run AI Prediction") and chosen_ticker:
    with st.spinner(f"AI Agent is analyzing {chosen_ticker}..."):
        try:
            ai_df, analysis = advanced_ai_prediction(chosen_ticker, days=horizon_days)
            
            current_data = yf.Ticker(chosen_ticker).history(period="1d")
            
            if not current_data.empty:
                current_price = current_data['Close'].iloc[-1]
                
                last_pred = ai_df['Predicted_Price'].iloc[-1]
                last_lower = ai_df['Lower_Bound'].iloc[-1]
                last_upper = ai_df['Upper_Bound'].iloc[-1]
                
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
                c1.metric("Trend", analysis["Trend"])
                c2.metric("Volatility", analysis["Volatility"])
                c3.metric("Confidence", analysis["Confidence_Score"])
                
                st.info(f"Recommendation: **{analysis['Recommendation']}**")
                
                fig_ai = go.Figure()
                fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Predicted_Price'], name='AI Prediction', line=dict(color='purple')))
                fig_ai.add_trace(go.Scatter(
                    x=ai_df.index, y=ai_df['Upper_Bound'],
                    fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
                ))
                fig_ai.add_trace(go.Scatter(
                    x=ai_df.index, y=ai_df['Lower_Bound'],
                    fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                    name='Confidence Interval', fillcolor='rgba(128, 0, 128, 0.2)'
                ))
                st.plotly_chart(fig_ai, use_container_width=True)
                
            else:
                st.error("Could not fetch current price for return calculation.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    if not chosen_ticker:
        st.info("Please select or enter at least one valid ticker to predict.")

# Back button
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")  # ← change to your actual main file name
