import streamlit as st
import yfinance as yf
from utils import advanced_ai_prediction
from difflib import get_close_matches

# ─── Reuse helpers from home page ──────────────────────────────────────────
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

# ─── Valuation functions ───────────────────────────────────────────────────
def peter_lynch_fair_value(eps, growth_pct, div_yield_pct=0):
    fair_pe = growth_pct + div_yield_pct
    fair_value = eps * fair_pe if eps > 0 else None
    return fair_value, fair_pe

def graham_intrinsic_value(eps, growth_pct, bond_yield_pct=7.6):
    if eps <= 0:
        return None, None
    base = 8.5 + 2 * growth_pct
    v = (eps * base * 4.4) / bond_yield_pct
    mos_30 = v * 0.70  # 30% margin of safety
    return v, mos_30

# ─── Page ──────────────────────────────────────────────────────────────────
st.title("🔮 AI Premium Prediction + Valuation")

# Premium check with fallback
email = st.session_state.get("premium_email", None)
if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai")

if not email or email not in st.session_state.premium_users:
    st.error("You need premium access to use this page. Go back to the main page and subscribe/verify.")
    if st.button("← Back to Main Portfolio"):
        st.switch_page("Home.py")
    st.stop()

st.success(f"Premium active for {email}")

# ─── Stock / ETF selector ──────────────────────────────────────────────────
st.markdown("### Select Stock(s) / ETF(s)")

search_options = load_search_options()

selected_assets = st.multiselect(
    "🔍 Search & select stocks / ETFs",
    options=search_options,
    key="ai_page_selected_assets"
)

manual_assets = st.text_input(
    "✍️ Or manually type names / tickers (comma separated)",
    "",
    key="ai_page_manual_assets"
)

user_assets = list(selected_assets) + [x.strip() for x in manual_assets.split(",") if x.strip()]

# Resolve and show valid tickers
valid_tickers = []
if user_assets:
    resolved = resolve_assets(user_assets)
    valid_tickers = [v for v in resolved.values() if v]
    if valid_tickers:
        st.write("Resolved tickers:", ", ".join(valid_tickers))
    else:
        st.warning("No valid tickers resolved.")

# Choose ticker for prediction (default first valid)
chosen_ticker = st.selectbox(
    "Ticker for AI Prediction & Valuation",
    options=valid_tickers if valid_tickers else ["RELIANCE.NS"],
    index=0 if valid_tickers else None
)

# Horizon
horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()), index=1)
horizon_days = horizon_map[horizon_label]

# ─── Run button ────────────────────────────────────────────────────────────
if st.button("Run AI Prediction + Valuation") and chosen_ticker:
    with st.spinner(f"Analyzing {chosen_ticker}..."):
        try:
            # AI Prediction
            ai_df, analysis = advanced_ai_prediction(chosen_ticker, days=horizon_days)
            
            current_data = yf.Ticker(chosen_ticker).history(period="1d")
            if current_data.empty:
                st.error("Could not fetch current price.")
                st.stop()
                
            current_price = current_data['Close'].iloc[-1]
            
            last_pred = ai_df['Predicted_Price'].iloc[-1]
            last_lower = ai_df['Lower_Bound'].iloc[-1]
            last_upper = ai_df['Upper_Bound'].iloc[-1]
            
            ret = (last_pred - current_price) / current_price
            ret_low = (last_lower - current_price) / current_price
            ret_high = (last_upper - current_price) / current_price
            
            st.metric("AI Predicted Return", f"{ret*100:.2f}%")
            st.write(f"Confidence Range: {ret_low*100:.2f}% to {ret_high*100:.2f}% (Horizon: {horizon_days} days)")
            
            st.markdown("### AI Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend", analysis["Trend"])
            c2.metric("Volatility", analysis["Volatility"])
            c3.metric("Confidence", analysis["Confidence_Score"])
            st.info(f"Recommendation: **{analysis['Recommendation']}**")
            
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Predicted_Price'], name='AI Prediction', line=dict(color='purple')))
            fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Upper_Bound'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df['Lower_Bound'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(128, 0, 128, 0.2)'))
            st.plotly_chart(fig_ai, use_container_width=True)

            # ─── Valuation Section ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Valuation Analysis (Peter Lynch + Benjamin Graham)")

            ticker_obj = yf.Ticker(chosen_ticker)
            info = ticker_obj.info

            eps = info.get('trailingEps', info.get('forwardEps', None))
            growth_pct = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 15  # fallback
            div_yield_pct = (info.get('dividendYield', 0) or 0) * 100
            current_pe = info.get('trailingPE', info.get('forwardPE', None))

            if eps is None or eps <= 0:
                st.warning("Could not fetch valid EPS data for valuation.")
            else:
                # Peter Lynch
                lynch_fv, lynch_fair_pe = peter_lynch_fair_value(eps, growth_pct, div_yield_pct)
                peg = current_pe / growth_pct if growth_pct > 0 and current_pe else None

                st.markdown("**Peter Lynch Valuation**")
                st.write(f"Fair P/E: **{lynch_fair_pe:.1f}x**")
                if lynch_fv:
                    st.write(f"Fair Value: **₹{lynch_fv:.0f}**")
                if peg:
                    st.write(f"PEG Ratio: **{peg:.2f}** → {'Undervalued' if peg < 1 else 'Fair' if peg <= 1.5 else 'Overvalued'}")

                # Benjamin Graham
                graham_v, graham_mos = graham_intrinsic_value(eps, growth_pct)

                st.markdown("**Benjamin Graham Valuation** (AAA yield ~7.6%)")
                if graham_v:
                    st.write(f"Intrinsic Value: **₹{graham_v:.0f}**")
                    st.write(f"With 30% Margin of Safety: Buy below **₹{graham_mos:.0f}**")

                # Comparison
                st.markdown("**Verdict vs Current Price (₹{:.0f})**".format(current_price))
                if lynch_fv and current_price < lynch_fv:
                    st.success("Undervalued according to Peter Lynch")
                elif lynch_fv:
                    st.warning("Fairly valued or overvalued on Lynch")

                if graham_mos and current_price < graham_mos:
                    st.success("Undervalued with safety margin (Graham)")
                elif graham_v:
                    st.warning("Above Graham margin of safety → consider overvalued")

        except Exception as e:
            st.error(f"Error during analysis: {e}")

# ─── Navigation ────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")
