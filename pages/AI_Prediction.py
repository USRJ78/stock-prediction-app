# pages/AI_Prediction.py
# ✅ Paste this file exactly into your Git repo under: pages/AI_Prediction.py
# Requires: streamlit, yfinance, pandas, numpy, plotly
# Also requires your existing: utils.py with advanced_ai_prediction()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from difflib import get_close_matches

from utils import advanced_ai_prediction


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (Search/Resolve)  — same idea as your Grok code but safer
# ─────────────────────────────────────────────────────────────────────────────
ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_stock_list():
    """
    Pulls NSE equity list. Sometimes NSE blocks requests.
    If it fails, we return an empty dict and app still works with manual tickers.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url)
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip() + ".NS"
        # NAME OF COMPANY is in NSE file
        return dict(zip(df["NAME OF COMPANY"].astype(str).str.upper().str.strip(), df["SYMBOL"]))
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def load_search_options():
    stock_map = load_nse_stock_list()
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

def _looks_like_symbol(s: str) -> bool:
    # e.g. RELIANCE, TCS, INFY, HDFCBANK etc.
    return s.replace("-", "").replace("&", "").isalnum()

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_assets(user_inputs):
    """
    Resolves company names → .NS tickers, supports ETFs, supports user-entered tickers.
    Safer than the original:
    - If user types RELIANCE -> RELIANCE.NS
    - If user types TCS.NS/BSE suffix -> keep
    """
    stock_map = load_nse_stock_list()
    resolved = {}

    for item in user_inputs:
        raw = str(item).strip()
        key = raw.upper().strip()

        if not key:
            continue

        # Already a ticker with suffix (e.g., RELIANCE.NS, SBIN.BO)
        if key.endswith(".NS") or key.endswith(".BO"):
            resolved[raw] = key
            continue

        # If it contains a dot but not standard suffix, still accept as-is (power users)
        if "." in key:
            resolved[raw] = key
            continue

        # ETF shortcuts
        if key in ETF_MAP:
            resolved[raw] = ETF_MAP[key]
            continue

        # If user typed a symbol-like string, auto-append .NS
        if _looks_like_symbol(key) and key.isalpha():
            resolved[raw] = f"{key}.NS"
            continue

        # Fuzzy match against company names from NSE list
        if stock_map:
            matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6)
            resolved[raw] = stock_map[matches[0]] if matches else None
        else:
            resolved[raw] = None

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Value Screener (Peter Lynch → Graham → MOS)
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan

def cagr(first, last, years):
    if years <= 0:
        return np.nan
    first = safe_float(first)
    last = safe_float(last)
    if not np.isfinite(first) or not np.isfinite(last) or first <= 0 or last <= 0:
        return np.nan
    return (last / first) ** (1 / years) - 1

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetches:
    - price
    - EPS (TTM)
    - PE
    - attempts growth estimate from annual earnings history
    """
    t = yf.Ticker(ticker)

    # info can be missing/partial for India
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    price = safe_float(info.get("currentPrice"))
    if not np.isfinite(price):
        price = safe_float(info.get("regularMarketPrice"))
    if not np.isfinite(price):
        price = safe_float(info.get("previousClose"))

    eps = safe_float(info.get("trailingEps"))
    pe = safe_float(info.get("trailingPE"))

    # Growth estimate: use earnings CAGR from yf.Ticker().earnings if available
    growth_pct = np.nan
    growth_src = ""

    try:
        earnings_df = t.earnings  # sometimes available with columns Revenue, Earnings (annual)
        if isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty and "Earnings" in earnings_df.columns:
            years = sorted(list(earnings_df.index))
            if len(years) >= 3:
                first_year = years[0]
                last_year = years[-1]
                span = int(last_year) - int(first_year)
                if span >= 2:
                    first_val = earnings_df.loc[first_year, "Earnings"]
                    last_val = earnings_df.loc[last_year, "Earnings"]
                    g = cagr(first_val, last_val, span)
                    if np.isfinite(g):
                        growth_pct = g * 100
                        growth_src = f"Earnings CAGR ({first_year}→{last_year})"
    except Exception:
        pass

    # fallback to quarterly growth field if exists
    if not np.isfinite(growth_pct):
        qg = safe_float(info.get("earningsQuarterlyGrowth"))
        if np.isfinite(qg):
            growth_pct = qg * 100
            growth_src = "earningsQuarterlyGrowth (YoY)"

    mcap = safe_float(info.get("marketCap"))

    return {
        "ticker": ticker,
        "price": price,
        "eps": eps,
        "pe": pe,
        "growth_pct": growth_pct,
        "growth_src": growth_src,
        "market_cap": mcap,
    }

def peter_lynch_screen(pe, growth_pct, peg_limit):
    """
    Simple Lynch-style:
    - PE > 0
    - Growth% > 0
    - PEG = PE / Growth% <= peg_limit
    """
    reasons = []
    passed = True

    if not np.isfinite(pe) or pe <= 0:
        passed = False
        reasons.append("P/E not available or ≤ 0")

    if not np.isfinite(growth_pct) or growth_pct <= 0:
        passed = False
        reasons.append("Growth% not available or ≤ 0")

    peg = np.nan
    if passed:
        peg = pe / growth_pct if growth_pct != 0 else np.nan
        if not np.isfinite(peg):
            passed = False
            reasons.append("PEG could not be computed")
        elif peg > peg_limit:
            passed = False
            reasons.append(f"PEG {peg:.2f} > limit {peg_limit}")

    return passed, peg, reasons

def graham_intrinsic_value(eps, growth_pct, bond_yield_pct):
    """
    Graham formula commonly used:
    Intrinsic = EPS * (8.5 + 2g) * (4.4 / Y)
    where g and Y are in percent.
    """
    if not np.isfinite(eps) or eps <= 0:
        return np.nan, "EPS not available or ≤ 0"
    if not np.isfinite(growth_pct) or growth_pct < 0:
        return np.nan, "Growth% not available or < 0"
    if not np.isfinite(bond_yield_pct) or bond_yield_pct <= 0:
        return np.nan, "Bond yield not valid"

    g = float(np.clip(growth_pct, 0, 25))  # guardrail
    Y = float(bond_yield_pct)

    intrinsic = eps * (8.5 + 2 * g) * (4.4 / Y)
    return intrinsic, f"Used g={g:.2f}% (capped 0–25), Y={Y:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Page UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Premium Prediction", layout="wide")
st.title("🔮 AI Premium Prediction")

# Premium check (safer)
email = st.session_state.get("premium_email", None)
if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai_page")

premium_users = st.session_state.get("premium_users", set())

if not email or email not in premium_users:
    st.error("Premium access required for this page. Please return to the main page and subscribe/verify.")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")  # change if your main file is different
    st.stop()

st.success(f"Premium active for {email}")

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

valid_tickers = []
resolved = {}

if user_assets:
    resolved = resolve_assets(user_assets)
    valid_tickers = [v for v in resolved.values() if v]
    if valid_tickers:
        st.write("Resolved tickers:", ", ".join(valid_tickers))
    else:
        st.warning("No valid tickers could be resolved from your selection. Try typing tickers like RELIANCE.NS")

# Choose the ticker to forecast (AI tab)
if valid_tickers:
    chosen_ticker = st.selectbox("Select asset for prediction", options=valid_tickers, index=0)
else:
    chosen_ticker = st.text_input("Enter ticker manually (fallback)", "RELIANCE.NS").upper().strip()

tabs = st.tabs(["🤖 AI Forecast", "📌 Lynch → Graham → MOS (Value Screener)"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: AI Forecast (your existing flow)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🤖 AI Forecast")

    horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()), index=1)
    horizon_days = horizon_map[horizon_label]

    if st.button("Run AI Prediction", key="run_ai_pred") and chosen_ticker:
        with st.spinner(f"AI Agent is analyzing {chosen_ticker}..."):
            try:
                ai_df, analysis = advanced_ai_prediction(chosen_ticker, days=horizon_days)

                current_data = yf.Ticker(chosen_ticker).history(period="1d")
                if current_data.empty:
                    st.error("Could not fetch current price for return calculation.")
                else:
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
                    c1.metric("Trend", str(analysis.get("Trend", "N/A")))
                    c2.metric("Volatility", str(analysis.get("Volatility", "N/A")))
                    c3.metric("Confidence", str(analysis.get("Confidence_Score", "N/A")))

                    st.info(f"Recommendation: **{analysis.get('Recommendation', 'N/A')}**")

                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Predicted_Price"],
                        name="AI Prediction"
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Upper_Bound"],
                        fill=None, mode="lines",
                        line_color="rgba(0,0,0,0)", showlegend=False
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Lower_Bound"],
                        fill="tonexty", mode="lines",
                        line_color="rgba(0,0,0,0)",
                        name="Confidence Interval", fillcolor="rgba(128, 0, 128, 0.2)"
                    ))
                    st.plotly_chart(fig_ai, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Value Screener (Lynch → Graham → MOS)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("📌 Lynch → Graham → MOS (Value Screener)")

    cA, cB, cC = st.columns([1.2, 1, 1])

    with cA:
        peg_limit = st.number_input("Peter Lynch PEG limit", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        mos_pct = st.number_input("Margin of Safety (%)", min_value=0, max_value=80, value=30, step=1)

    with cB:
        bond_yield = st.number_input(
            "Bond yield used in Graham formula (%)",
            min_value=0.5, max_value=20.0, value=8.0, step=0.1,
            help="Set an India-appropriate AAA/Corporate bond yield that you trust."
        )

    with cC:
        default_growth = st.number_input(
            "Default growth% (only if missing)",
            min_value=0.0, max_value=30.0, value=12.0, step=0.5,
            help="If Yahoo doesn’t provide earnings history/growth, this is used."
        )

    # Use all tickers user selected/typed (not just chosen_ticker)
    tickers_for_value = valid_tickers if valid_tickers else ([chosen_ticker] if chosen_ticker else [])

    if not tickers_for_value:
        st.info("Select or type tickers above, then run the screener.")
    else:
        st.markdown("#### Optional: Override growth% per ticker (recommended)")
        override_df = pd.DataFrame({"ticker": tickers_for_value, "growth_override_pct": [np.nan] * len(tickers_for_value)})
        edited = st.data_editor(
            override_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "growth_override_pct": st.column_config.NumberColumn(
                    "growth_override_pct",
                    min_value=0.0, max_value=30.0, step=0.5, format="%.2f",
                    help="If entered, overrides growth% for that ticker."
                )
            },
        )

        override_map = {}
        if edited is not None and not edited.empty:
            for _, r in edited.iterrows():
                tk = str(r.get("ticker", "")).strip()
                gv = safe_float(r.get("growth_override_pct"))
                if tk and np.isfinite(gv):
                    override_map[tk] = gv

        if st.button("Run Value Screener", key="run_value"):
            rows = []
            with st.spinner("Fetching fundamentals and calculating value..."):
                for tk in tickers_for_value:
                    f = fetch_fundamentals(tk)
                    price = f["price"]
                    eps = f["eps"]
                    pe = f["pe"]

                    # choose growth: override > fetched > default
                    growth_pct = override_map.get(tk, np.nan)
                    growth_src = "Manual override"
                    if not np.isfinite(growth_pct):
                        growth_pct = f["growth_pct"]
                        growth_src = f["growth_src"] if f["growth_src"] else "Yahoo/earnings (unknown)"
                    if not np.isfinite(growth_pct):
                        growth_pct = float(default_growth)
                        growth_src = "Default growth input"

                    lynch_pass, peg, lynch_reasons = peter_lynch_screen(pe, growth_pct, peg_limit)
                    intrinsic, graham_note = graham_intrinsic_value(eps, growth_pct, bond_yield)

                    mos_price = np.nan
                    verdict = "N/A"
                    mos_gap_pct = np.nan

                    if np.isfinite(intrinsic):
                        mos_price = intrinsic * (1 - mos_pct / 100.0)

                    if np.isfinite(price) and np.isfinite(mos_price) and mos_price > 0:
                        mos_gap_pct = (mos_price - price) / price * 100.0
                        verdict = "Undervalued ✅" if price < mos_price else "Overvalued / No MOS ❌"

                    rows.append({
                        "Ticker": tk,
                        "Current Price": price,
                        "P/E": pe,
                        "EPS (TTM)": eps,
                        "Growth % (g)": growth_pct,
                        "Growth Source": growth_src,
                        "PEG": peg,
                        "Lynch Screen": "PASS" if lynch_pass else "FAIL",
                        "Lynch Reasons": "; ".join(lynch_reasons) if lynch_reasons else "",
                        "Graham Intrinsic": intrinsic,
                        f"MOS Price ({mos_pct}%)": mos_price,
                        "Verdict": verdict,
                        "MOS Gap % (MOS - Price)": mos_gap_pct,
                        "Graham Note": graham_note,
                    })

            df = pd.DataFrame(rows)
            st.markdown("### Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("### Shortlist (Undervalued ✅ + Lynch PASS)")
            shortlist = df[(df["Verdict"] == "Undervalued ✅") & (df["Lynch Screen"] == "PASS")].copy()
            if shortlist.empty:
                st.info("No stocks met both conditions with current inputs.")
            else:
                shortlist = shortlist.sort_values("MOS Gap % (MOS - Price)", ascending=False)
                st.dataframe(
                    shortlist[[
                        "Ticker", "Current Price", "P/E", "Growth % (g)", "PEG",
                        "Graham Intrinsic", f"MOS Price ({mos_pct}%)", "MOS Gap % (MOS - Price)"
                    ]],
                    use_container_width=True
                )

            st.markdown("### Notes")
            st.write(
                "- For many Indian stocks, **growth% may be missing** in Yahoo data. Use overrides for better accuracy.\n"
                "- Graham intrinsic value is very sensitive to **g** and **bond yield**.\n"
                "- This is a **screener** to support decisions, not financial advice."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")  # change if your main file is different
