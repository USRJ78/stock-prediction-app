# pages/AI_Prediction.py
# ✅ FULL UPDATED FILE (paste & replace)
# ✅ NSE list robust (LIVE NSE -> fallback ../data/EQUITY_L.csv)
# ✅ Value Picks results persist after reruns (amount sliders etc won't reset)
# ✅ Auto Value Picks supports: Peter Lynch / Benjamin Graham / EPS×Multiple / Any combo / All
# ✅ Portfolio Builder target can use Graham MOS OR EPS×M MOS

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import seaborn as sns

from utils import advanced_ai_prediction
from auth_store import load_premium_users

import requests
from io import StringIO
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Page config (ONLY ONCE)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Premium Prediction", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Persist screening results across reruns (FIX for "goes back to before screening")
# ─────────────────────────────────────────────────────────────────────────────
if "screen_done" not in st.session_state:
    st.session_state.screen_done = False
if "screen_df" not in st.session_state:
    st.session_state.screen_df = None
if "screen_shortlist" not in st.session_state:
    st.session_state.screen_shortlist = None
if "screen_inputs" not in st.session_state:
    st.session_state.screen_inputs = {}


# ─────────────────────────────────────────────────────────────────────────────
# NSE list + resolver helpers
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
    ✅ Robust NSE loader:
    1) Try LIVE NSE CSV with headers
    2) Fallback to repo file: ../data/EQUITY_L.csv (since this file is in pages/)
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

    # 2) LOCAL fallback: repo_root/data/EQUITY_L.csv
    try:
        base_dir = Path(__file__).resolve().parent        # .../pages
        csv_path = base_dir.parent / "data" / "EQUITY_L.csv"
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

def _looks_like_symbol(s: str) -> bool:
    s = s.replace("-", "").replace("&", "")
    return s.isalnum()

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_list() or {}
    resolved = {}
    for item in user_inputs:
        raw = str(item).strip()
        key = raw.upper().strip()
        if not key:
            continue

        if key.endswith(".NS") or key.endswith(".BO"):
            resolved[raw] = key
            continue

        if "." in key:
            resolved[raw] = key
            continue

        if key in ETF_MAP:
            resolved[raw] = ETF_MAP[key]
            continue

        if _looks_like_symbol(key) and key.isalpha():
            resolved[raw] = f"{key}.NS"
            continue

        matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6) if stock_map else []
        resolved[raw] = stock_map[matches[0]] if matches else None

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Growth: ONLY 5Y CAGR (Profit preferred, Sales fallback) — NO quarterly growth
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def compute_cagr_5y(series: pd.Series) -> float:
    """
    Strict 5Y CAGR (needs ~6 annual points).
    Returns CAGR in percent.
    """
    series = series.dropna()
    if len(series) < 6:
        return np.nan

    latest = safe_float(series.iloc[0])   # yfinance usually newest -> oldest
    oldest = safe_float(series.iloc[-1])
    years = len(series) - 1

    if not np.isfinite(latest) or not np.isfinite(oldest) or latest <= 0 or oldest <= 0:
        return np.nan

    return ((latest / oldest) ** (1 / years) - 1) * 100

def get_growth_5y_cagr(ticker: str, default_growth: float):
    """
    Returns (growth_pct, source_label)
    Growth is strictly 5Y CAGR:
      1) Net Income CAGR (Profit)
      2) Total Revenue CAGR (Sales)
      3) default fallback
    """
    t = yf.Ticker(ticker)

    try:
        fin = t.financials  # annual income statement
    except Exception:
        fin = None

    # 1) Profit CAGR
    try:
        if isinstance(fin, pd.DataFrame) and "Net Income" in fin.index:
            g = compute_cagr_5y(fin.loc["Net Income"])
            if np.isfinite(g) and g > 0:
                return float(g), "5Y Profit CAGR"
    except Exception:
        pass

    # 2) Sales CAGR
    try:
        if isinstance(fin, pd.DataFrame) and "Total Revenue" in fin.index:
            g = compute_cagr_5y(fin.loc["Total Revenue"])
            if np.isfinite(g) and g > 0:
                return float(g), "5Y Sales CAGR"
    except Exception:
        pass

    return float(default_growth), "Default growth (fallback)"


# ─────────────────────────────────────────────────────────────────────────────
# Fundamentals + Lynch + Graham + EPS×Multiple (separate)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch price, EPS, PE, market cap from Yahoo.
    Growth is computed separately via get_growth_5y_cagr().
    """
    t = yf.Ticker(ticker)
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
    mcap = safe_float(info.get("marketCap"))

    return {"ticker": ticker, "price": price, "eps": eps, "pe": pe, "market_cap": mcap}

def peter_lynch_screen(pe, growth_pct, peg_limit):
    reasons = []
    passed = True

    if not np.isfinite(pe) or pe <= 0:
        passed = False
        reasons.append("P/E not available or ≤ 0")

    if not np.isfinite(growth_pct) or growth_pct <= 0:
        passed = False
        reasons.append("5Y CAGR not available or ≤ 0")

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
    Classic Graham (approx):
    Intrinsic = EPS * (8.5 + 2g) * (4.4 / Y)
    g and Y are in percent.
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

def eps_multiple_target(eps: float, multiple: float):
    """
    Simple target price:
      Target = EPS * multiple
    """
    if not np.isfinite(eps) or eps <= 0:
        return np.nan, "EPS not available or ≤ 0"
    if not np.isfinite(multiple) or multiple <= 0:
        return np.nan, "Multiple not valid"
    return float(eps) * float(multiple), f"Used EPS×M where M={float(multiple):.0f}"

def compute_value_row(
    tk: str,
    peg_limit: float,
    mos_pct: float,
    bond_yield: float,
    default_growth: float,
    eps_mult: float
) -> dict:
    f = fetch_fundamentals(tk)
    price = f["price"]
    eps = f["eps"]
    pe = f["pe"]
    mcap = f["market_cap"]

    growth_pct, growth_src = get_growth_5y_cagr(tk, default_growth)

    # Lynch (PEG)
    lynch_pass, peg, lynch_reasons = peter_lynch_screen(pe, growth_pct, peg_limit)

    # Graham intrinsic
    intrinsic, graham_note = graham_intrinsic_value(eps, growth_pct, bond_yield)

    mos_price = np.nan
    verdict = "N/A"
    mos_gap_pct = np.nan
    if np.isfinite(intrinsic):
        mos_price = intrinsic * (1 - mos_pct / 100.0)
    if np.isfinite(price) and np.isfinite(mos_price) and mos_price > 0:
        mos_gap_pct = (mos_price - price) / price * 100.0
        verdict = "Undervalued ✅" if price < mos_price else "Overvalued / No MOS ❌"

    # EPS×Multiple (separate)
    eps_target, eps_note = eps_multiple_target(eps, eps_mult)

    eps_mos_price = np.nan
    eps_verdict = "N/A"
    eps_gap_pct = np.nan
    if np.isfinite(eps_target):
        eps_mos_price = eps_target * (1 - mos_pct / 100.0)
    if np.isfinite(price) and np.isfinite(eps_mos_price) and eps_mos_price > 0:
        eps_gap_pct = (eps_mos_price - price) / price * 100.0
        eps_verdict = "Undervalued ✅" if price < eps_mos_price else "Overvalued / No MOS ❌"

    return {
        "Ticker": tk,
        "Mkt Cap": mcap,
        "Current Price": price,
        "P/E": pe,
        "EPS (TTM)": eps,
        "Growth % (g)": growth_pct,
        "Growth Source": growth_src,

        # Lynch
        "PEG": peg,
        "Lynch Screen": "PASS" if lynch_pass else "FAIL",
        "Lynch Reasons": "; ".join(lynch_reasons) if lynch_reasons else "",

        # Graham
        "Graham Intrinsic": intrinsic,
        f"MOS Price ({mos_pct}%)": mos_price,
        "Verdict": verdict,
        "MOS Gap % (MOS - Price)": mos_gap_pct,
        "Graham Note": graham_note,

        # EPS×Multiple
        "EPS Target (EPS×M)": eps_target,
        f"EPS×M MOS Price ({mos_pct}%)": eps_mos_price,
        "EPS×M Verdict": eps_verdict,
        "EPS×M Gap % (MOS - Price)": eps_gap_pct,
        "EPS×M Note": eps_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Correlation/portfolio helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices_for_corr(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

def pick_least_correlated(corr_df: pd.DataFrame, n: int):
    """
    Greedy selection:
    - start with lowest-correlation pair
    - add next ticker that minimizes avg correlation to chosen set
    """
    tickers = list(corr_df.columns)
    if len(tickers) <= n:
        return tickers

    corr_vals = corr_df.copy()
    np.fill_diagonal(corr_vals.values, np.nan)

    i, j = np.unravel_index(np.nanargmin(corr_vals.values), corr_vals.shape)
    chosen = [tickers[i], tickers[j]]

    while len(chosen) < min(n, len(tickers)):
        remaining = [t for t in tickers if t not in chosen]
        best_t, best_score = None, 1e9
        for t in remaining:
            score = np.nanmean([corr_df.loc[t, c] for c in chosen])
            if score < best_score:
                best_score = score
                best_t = t
        chosen.append(best_t)

    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Page UI + Premium gate
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 AI Premium Prediction")

email = st.session_state.get("premium_email", None)
if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai_page").strip()

premium_users = load_premium_users()

if not email or email.lower().strip() not in premium_users:
    st.error("Premium access required for this page. Please return to the main page and subscribe/verify.")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")
    st.stop()

st.session_state["premium_email"] = email.lower().strip()
st.success(f"Premium active for {st.session_state['premium_email']}")

tabs = st.tabs(["🤖 AI Forecast", "✨ Auto Value Picks + Portfolio Builder"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: AI Forecast (kept)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🤖 AI Forecast")
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
    if user_assets:
        resolved = resolve_assets(user_assets)
        valid_tickers = [v for v in resolved.values() if v]
        if valid_tickers:
            st.write("Resolved tickers:", ", ".join(valid_tickers))
        else:
            st.warning("No valid tickers could be resolved from your selection.")

    if valid_tickers:
        chosen_ticker = st.selectbox("Select asset for prediction", options=valid_tickers, index=0)
    else:
        chosen_ticker = st.text_input("Enter ticker manually (fallback)", "RELIANCE.NS").upper().strip()

    horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    horizon_label = st.selectbox("Prediction Horizon", list(horizon_map.keys()), index=1)
    horizon_days = horizon_map[horizon_label]

    if st.button("Run AI Prediction", key="run_ai_pred") and chosen_ticker:
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

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend", str(analysis.get("Trend", "N/A")))
                    c2.metric("Volatility", str(analysis.get("Volatility", "N/A")))
                    c3.metric("Confidence", str(analysis.get("Confidence_Score", "N/A")))
                    st.info(f"Recommendation: **{analysis.get('Recommendation', 'N/A')}**")

                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Predicted_Price"], name="AI Prediction",
                        line=dict(color="purple")
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Upper_Bound"],
                        fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False
                    ))
                    fig_ai.add_trace(go.Scatter(
                        x=ai_df.index, y=ai_df["Lower_Bound"],
                        fill="tonexty", mode="lines", line_color="rgba(0,0,0,0)",
                        name="Confidence Interval", fillcolor="rgba(128, 0, 128, 0.2)"
                    ))
                    st.plotly_chart(fig_ai, use_container_width=True)
                else:
                    st.error("Could not fetch current price for return calculation.")
            except Exception as e:
                st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Auto Value Picks + Portfolio Builder
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("✨ Auto Value Picks + Portfolio Builder")
    st.caption("Growth (g) uses 5Y CAGR: Profit preferred, Sales fallback. Valuation can use Lynch, Graham, EPS×M, or combos.")

    # Screening inputs
    cA, cB, cC, cD = st.columns([1.0, 1.0, 1.0, 1.0])
    with cA:
        peg_limit = st.number_input("Lynch PEG limit", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="v_peg")
    with cB:
        mos_pct = st.number_input("Margin of Safety (%)", min_value=0, max_value=80, value=30, step=1, key="v_mos")
    with cC:
        bond_yield = st.number_input(
            "Bond yield used in Graham formula (%)",
            min_value=0.5, max_value=20.0, value=8.0, step=0.1, key="v_yield"
        )
    with cD:
        default_growth = st.number_input(
            "Default growth% (fallback if 5Y CAGR missing)",
            min_value=0.0, max_value=30.0, value=12.0, step=0.5, key="v_defg"
        )

    methods = st.multiselect(
        "Select valuation / screening methods",
        ["Peter Lynch (PEG)", "Benjamin Graham (Intrinsic)", "EPS Multiple (EPS×M)"],
        default=["Peter Lynch (PEG)", "Benjamin Graham (Intrinsic)"],
        key="v_methods"
    )

    eps_mult = st.slider(
        "EPS Multiple (M) for EPS×M method",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        key="v_eps_mult",
        help="Target price = EPS × M. MOS is applied to this target to decide undervalued/overvalued."
    )

    # Universe controls
    u1, u2 = st.columns([1, 1])
    with u1:
        universe_size = st.slider("Universe size (Top by market cap)", 100, 700, 300, 50, key="v_univ")
    with u2:
        top_n = st.slider("Show top N picks", 10, 100, 30, 5, key="v_topn")

    # Optional Market cap filter (₹ Crore)
    mc1, mc2 = st.columns([1, 1])
    with mc1:
        min_mcap_cr = st.number_input("Min Market Cap (₹ Crore)", min_value=0, value=0, step=100, key="v_min_mcap_cr")
    with mc2:
        max_mcap_cr = st.number_input("Max Market Cap (₹ Crore)", min_value=0, value=0, step=100, key="v_max_mcap_cr",
                                      help="0 = no upper limit")

    min_mcap = float(min_mcap_cr) * 1e7
    max_mcap = float(max_mcap_cr) * 1e7 if float(max_mcap_cr) > 0 else np.inf

    # Build NSE universe tickers
    stock_map = load_nse_stock_list()
    all_tickers = list(stock_map.values())

    if not all_tickers:
        st.error(
            "Could not load NSE equity list.\n\n"
            "✅ Ensure `data/EQUITY_L.csv` exists in your repo root\n"
            "✅ Streamlit Cloud → Manage app → Clear cache → Restart"
        )
        st.stop()

    run_clicked = st.button("Run Auto Screening", type="primary", key="run_value_screen")

    if run_clicked:
        max_workers = 24

        st.session_state.screen_inputs = dict(
            peg_limit=peg_limit,
            mos_pct=mos_pct,
            bond_yield=bond_yield,
            default_growth=default_growth,
            universe_size=universe_size,
            top_n=top_n,
            methods=methods,
            eps_mult=eps_mult,
            min_mcap_cr=min_mcap_cr,
            max_mcap_cr=max_mcap_cr
        )

        # Step 1: Top by market cap
        with st.spinner("Step 1/3: Selecting universe by market cap..."):
            mcap_rows = []

            def mcap_only(tk):
                f = fetch_fundamentals(tk)
                return tk, f.get("market_cap", np.nan)

            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for tk in all_tickers[:2000]:
                    futures.append(ex.submit(mcap_only, tk))
                for fut in as_completed(futures):
                    tk, mc = fut.result()
                    if np.isfinite(mc) and mc > 0:
                        mcap_rows.append((tk, mc))

            if not mcap_rows:
                st.error("Could not fetch market caps right now (Yahoo may be throttling). Try again.")
                st.stop()

            mcap_df = pd.DataFrame(mcap_rows, columns=["Ticker", "MktCap"]).sort_values("MktCap", ascending=False)

            # Apply market cap filter
            mcap_df = mcap_df[(mcap_df["MktCap"] >= min_mcap) & (mcap_df["MktCap"] <= max_mcap)]
            if mcap_df.empty:
                st.error("No stocks match the Market Cap filter. Reduce min cap / increase max cap.")
                st.stop()

            universe = mcap_df["Ticker"].head(universe_size).tolist()

        # Step 2: Run metrics for selected universe
        with st.spinner("Step 2/3: Computing Lynch/Graham/EPS×M (with 5Y CAGR)..."):
            rows = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures2 = [
                    ex.submit(compute_value_row, tk, peg_limit, mos_pct, bond_yield, default_growth, eps_mult)
                    for tk in universe
                ]
                for fut in as_completed(futures2):
                    rows.append(fut.result())

            df = pd.DataFrame(rows)

        # Step 3: Apply chosen filters (combo/all)
        shortlist = df.copy()

        if not methods:
            shortlist = df.iloc[0:0].copy()
        else:
            if "Peter Lynch (PEG)" in methods:
                shortlist = shortlist[shortlist["Lynch Screen"] == "PASS"]
            if "Benjamin Graham (Intrinsic)" in methods:
                shortlist = shortlist[shortlist["Verdict"] == "Undervalued ✅"]
            if "EPS Multiple (EPS×M)" in methods:
                shortlist = shortlist[shortlist["EPS×M Verdict"] == "Undervalued ✅"]

        # Sort preference
        if "Benjamin Graham (Intrinsic)" in methods and "MOS Gap % (MOS - Price)" in shortlist.columns:
            shortlist = shortlist.sort_values("MOS Gap % (MOS - Price)", ascending=False)
        elif "EPS Multiple (EPS×M)" in methods and "EPS×M Gap % (MOS - Price)" in shortlist.columns:
            shortlist = shortlist.sort_values("EPS×M Gap % (MOS - Price)", ascending=False)
        else:
            shortlist = shortlist.sort_values("Mkt Cap", ascending=False)

        # Persist results
        st.session_state.screen_df = df
        st.session_state.screen_shortlist = shortlist
        st.session_state.screen_done = True

    # Render results if already screened
    if st.session_state.screen_done:
        df = st.session_state.screen_df
        shortlist = st.session_state.screen_shortlist

        st.success("Auto screening complete ✅")

        colR1, colR2 = st.columns([1, 5])
        with colR1:
            if st.button("Reset screening"):
                st.session_state.screen_done = False
                st.session_state.screen_df = None
                st.session_state.screen_shortlist = None
                st.rerun()

        if shortlist is None or shortlist.empty:
            st.info("No stocks matched your selected method(s) + filters. Try relaxing PEG/MOS/market-cap filters or selecting fewer methods.")
            if df is not None:
                st.dataframe(df.sort_values("Mkt Cap", ascending=False), use_container_width=True)
            st.stop()

        # Show shortlist columns depending on methods
        show_cols = ["Ticker", "Current Price", "Mkt Cap", "EPS (TTM)", "P/E", "Growth % (g)", "Growth Source"]

        if "Peter Lynch (PEG)" in methods:
            show_cols += ["PEG", "Lynch Screen"]
        if "Benjamin Graham (Intrinsic)" in methods:
            show_cols += ["Graham Intrinsic", f"MOS Price ({mos_pct}%)", "Verdict", "MOS Gap % (MOS - Price)"]
        if "EPS Multiple (EPS×M)" in methods:
            show_cols += ["EPS Target (EPS×M)", f"EPS×M MOS Price ({mos_pct}%)", "EPS×M Verdict", "EPS×M Gap % (MOS - Price)"]

        show_cols = [c for c in show_cols if c in shortlist.columns]

        st.markdown("### ✅ Shortlist (based on selected method(s))")
        st.dataframe(shortlist.head(top_n)[show_cols], use_container_width=True)

        st.markdown("### 📄 Full scan results")
        st.dataframe(df.sort_values("Mkt Cap", ascending=False), use_container_width=True)

        # ============================
        # Portfolio Builder
        # ============================
        st.markdown("---")
        st.subheader("📦 Portfolio Builder (Least Correlated + Monte Carlo)")
        st.caption("Pick least-correlated basket, then estimate target portfolio value if prices reach chosen MOS target.")

        target_mode = st.selectbox(
            "Portfolio target to use",
            ["Graham MOS", "EPS×M MOS"],
            index=0,
            key="pb_target_mode"
        )

        if target_mode == "Graham MOS":
            mos_col = f"MOS Price ({mos_pct}%)"
            verdict_col = "Verdict"
        else:
            mos_col = f"EPS×M MOS Price ({mos_pct}%)"
            verdict_col = "EPS×M Verdict"

        # Inputs
        pA, pB, pC, pD = st.columns([1, 1, 1, 1])
        with pA:
            invest_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000, key="pb_amt")
        with pB:
            corr_lookback = st.selectbox("Correlation Lookback", ["6mo", "1y", "2y"], index=1, key="pb_lb")
        with pC:
            pick_n = st.slider("Least-correlated stocks to pick", min_value=3, max_value=15, value=6, step=1, key="pb_n")
        with pD:
            rf_rate = st.number_input("Risk-free rate (annual, %)", min_value=0.0, max_value=20.0, value=0.0, step=0.25, key="pb_rf")

        # Candidates from shortlist must have valid MOS target and price
        candidates = shortlist[["Ticker", "Current Price", mos_col, verdict_col]].dropna()
        candidates = candidates[(candidates["Current Price"] > 0) & (candidates[mos_col] > 0)]

        if len(candidates) < 3:
            st.warning("Need at least 3 stocks with valid target MOS price for the chosen target mode.")
            st.stop()

        cand_tickers = candidates["Ticker"].tolist()
        prices_corr = load_prices_for_corr(cand_tickers, corr_lookback)

        if prices_corr.empty or prices_corr.shape[1] < 3:
            st.error("Not enough price history to compute correlations. Try a longer lookback (1y/2y).")
            st.stop()

        rets = prices_corr.pct_change().dropna()
        corr = rets.corr()

        st.markdown("### 🔥 Correlation Heatmap (Candidates)")
        fig_hm = plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap="RdYlGn", center=0)
        st.pyplot(fig_hm)
        plt.close()

        selected_tickers = pick_least_correlated(corr, int(pick_n))
        st.markdown("### ✅ Selected Least-Correlated Basket")
        st.write(", ".join(selected_tickers))

        basket = candidates[candidates["Ticker"].isin(selected_tickers)].copy().set_index("Ticker")
        basket = basket.loc[selected_tickers]

        basket["Target Return (%)"] = ((basket[mos_col] / basket["Current Price"]) - 1) * 100
        st.markdown("### 🎯 Upside if Each Stock Reaches Target MOS Price")
        st.dataframe(basket[["Current Price", mos_col, "Target Return (%)"]], use_container_width=True)

        rets_sel = rets[selected_tickers].dropna()
        if rets_sel.empty:
            st.error("No aligned return data for selected tickers. Try increasing lookback.")
            st.stop()

        cov_daily = rets_sel.cov().values
        target_multiplier = (basket[mos_col].values / basket["Current Price"].values)
        exp_upside = basket["Target Return (%)"].values / 100.0

        st.markdown("### 💰 Example Allocation (Random Weights) + Target Portfolio Value")
        rand_w = np.random.random(len(selected_tickers))
        rand_w = rand_w / rand_w.sum()
        alloc_amounts = invest_amount * rand_w

        target_value = float(np.sum(alloc_amounts * target_multiplier))
        target_gain = (target_value / invest_amount - 1) * 100

        alloc_df = pd.DataFrame({
            "Ticker": selected_tickers,
            "Weight": rand_w,
            "Allocation (₹)": alloc_amounts,
            "Current Price": basket["Current Price"].values,
            "Target MOS Price": basket[mos_col].values,
            "Target Multiplier": target_multiplier
        })
        st.dataframe(alloc_df, use_container_width=True)
        st.metric("Target Portfolio Value if All Reach Target", f"₹{target_value:,.0f}", f"{target_gain:.2f}%")

        st.markdown("### 🎯 Monte Carlo Optimization (Max Sharpe-like using Target Upside)")
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            mc_sims = st.number_input("Simulations", min_value=500, max_value=50000, value=8000, step=500, key="pb_sims")
        with m2:
            annualize_vol = st.checkbox("Annualize volatility (×√252)", value=True, key="pb_ann")
        with m3:
            show_top = st.number_input("Show top portfolios", min_value=5, max_value=50, value=10, step=1, key="pb_top")

        rf = (rf_rate / 100.0)
        vol_scale = np.sqrt(252) if annualize_vol else 1.0

        results = []
        weight_store = []

        for _ in range(int(mc_sims)):
            w = np.random.random(len(selected_tickers))
            w = w / w.sum()
            weight_store.append(w)

            port_ret = float(np.dot(w, exp_upside))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_daily, w)))) * vol_scale
            sharpe_like = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
            results.append([port_ret, port_vol, sharpe_like])

        sim_df = pd.DataFrame(results, columns=["Target_Upside_Return", "Volatility", "Sharpe_Like"])
        sim_df = sim_df.replace([np.inf, -np.inf], np.nan).dropna()

        if sim_df.empty:
            st.error("Monte Carlo produced no valid portfolios. Try a different basket/lookback.")
            st.stop()

        best_idx = sim_df["Sharpe_Like"].idxmax()
        best_w = weight_store[int(best_idx)]

        best_alloc = pd.DataFrame({
            "Ticker": selected_tickers,
            "Weight": best_w,
            "Allocation (₹)": invest_amount * best_w
        }).sort_values("Weight", ascending=False)

        st.markdown("#### ✅ Best Portfolio (Max Sharpe-like)")
        st.dataframe(best_alloc, use_container_width=True)

        best_target_value = float(np.sum((invest_amount * best_w) * target_multiplier))
        best_gain = (best_target_value / invest_amount - 1) * 100
        st.metric("Best Target Portfolio Value at Target", f"₹{best_target_value:,.0f}", f"{best_gain:.2f}%")

        fig_mc = px.scatter(
            sim_df,
            x="Volatility",
            y="Target_Upside_Return",
            color="Sharpe_Like",
            hover_data=["Sharpe_Like"]
        )
        fig_mc.add_trace(go.Scatter(
            x=[sim_df.loc[best_idx, "Volatility"]],
            y=[sim_df.loc[best_idx, "Target_Upside_Return"]],
            mode="markers",
            name="Best (Max Sharpe-like)",
            marker=dict(size=18, color="red")
        ))
        fig_mc.update_layout({"plot_bgcolor": "white"})
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown(f"### Top {int(show_top)} Portfolios")
        st.dataframe(sim_df.sort_values("Sharpe_Like", ascending=False).head(int(show_top)), use_container_width=True)

    else:
        st.info("Click **Run Auto Screening** to generate value picks and unlock the portfolio builder below.")

# Navigation
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")
