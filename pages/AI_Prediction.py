# pages/AI_Prediction.py
# ✅ Paste exactly into Git repo at: pages/AI_Prediction.py
# pip requirements: streamlit yfinance pandas numpy plotly

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import advanced_ai_prediction  # keep your existing utils


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: NSE list + resolver
# ─────────────────────────────────────────────────────────────────────────────
ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_equity_df():
    """
    NSE equity list. Sometimes NSE blocks requests; we fallback to empty.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url)
        # Normalize columns
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip()
        df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.upper().str.strip()
        df["TICKER"] = df["SYMBOL"] + ".NS"
        return df[["NAME OF COMPANY", "SYMBOL", "TICKER"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["NAME OF COMPANY", "SYMBOL", "TICKER"])

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_stock_map():
    df = load_nse_equity_df()
    return dict(zip(df["NAME OF COMPANY"], df["TICKER"]))

@st.cache_data(ttl=3600, show_spinner=False)
def load_search_options():
    stock_map = load_nse_stock_map()
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

def _looks_like_symbol(s: str) -> bool:
    s = s.replace("-", "").replace("&", "")
    return s.isalnum()

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_map()
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

        # If user typed a symbol-like code, append .NS
        if _looks_like_symbol(key) and key.isalpha():
            resolved[raw] = f"{key}.NS"
            continue

        # Fuzzy match company name
        matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6) if stock_map else []
        resolved[raw] = stock_map[matches[0]] if matches else None

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Value Screener: Lynch → Graham → MOS
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
    Fetch price, EPS, PE, market cap, and try to estimate growth using earnings CAGR.
    Works best when Yahoo provides earnings data (varies for India).
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

    growth_pct = np.nan
    growth_src = ""

    # Try annual earnings CAGR
    try:
        e = t.earnings
        if isinstance(e, pd.DataFrame) and not e.empty and "Earnings" in e.columns:
            years = sorted(list(e.index))
            if len(years) >= 3:
                first_year = years[0]
                last_year = years[-1]
                span = int(last_year) - int(first_year)
                if span >= 2:
                    g = cagr(e.loc[first_year, "Earnings"], e.loc[last_year, "Earnings"], span)
                    if np.isfinite(g):
                        growth_pct = g * 100
                        growth_src = f"Earnings CAGR ({first_year}→{last_year})"
    except Exception:
        pass

    # Fallback: quarterly growth (rarely populated for India)
    if not np.isfinite(growth_pct):
        qg = safe_float(info.get("earningsQuarterlyGrowth"))
        if np.isfinite(qg):
            growth_pct = qg * 100
            growth_src = "earningsQuarterlyGrowth (YoY)"

    return {
        "ticker": ticker,
        "price": price,
        "eps": eps,
        "pe": pe,
        "market_cap": mcap,
        "growth_pct": growth_pct,
        "growth_src": growth_src,
    }

def peter_lynch_screen(pe, growth_pct, peg_limit):
    reasons = []
    passed = True

    if not np.isfinite(pe) or pe <= 0:
        passed = False
        reasons.append("P/E missing or ≤ 0")

    if not np.isfinite(growth_pct) or growth_pct <= 0:
        passed = False
        reasons.append("Growth% missing or ≤ 0")

    peg = np.nan
    if passed:
        peg = pe / growth_pct if growth_pct != 0 else np.nan
        if not np.isfinite(peg):
            passed = False
            reasons.append("PEG could not be computed")
        elif peg > peg_limit:
            passed = False
            reasons.append(f"PEG {peg:.2f} > {peg_limit}")

    return passed, peg, reasons

def graham_intrinsic_value(eps, growth_pct, bond_yield_pct):
    """
    Intrinsic = EPS * (8.5 + 2g) * (4.4 / Y)
    g and Y are in percent.
    """
    if not np.isfinite(eps) or eps <= 0:
        return np.nan, "EPS missing or ≤ 0"
    if not np.isfinite(growth_pct) or growth_pct < 0:
        return np.nan, "Growth% missing or < 0"
    if not np.isfinite(bond_yield_pct) or bond_yield_pct <= 0:
        return np.nan, "Bond yield invalid"

    g = float(np.clip(growth_pct, 0, 25))  # guardrail
    Y = float(bond_yield_pct)

    intrinsic = eps * (8.5 + 2 * g) * (4.4 / Y)
    return intrinsic, f"Used g={g:.2f}% (capped 0–25), Y={Y:.2f}%"

def compute_value_row(tk, peg_limit, mos_pct, bond_yield, default_growth):
    f = fetch_fundamentals(tk)
    price = f["price"]
    eps = f["eps"]
    pe = f["pe"]
    mcap = f["market_cap"]

    growth = f["growth_pct"]
    growth_src = f["growth_src"] if f["growth_src"] else "Yahoo (missing fields)"
    if not np.isfinite(growth):
        growth = float(default_growth)
        growth_src = "Default growth input"

    lynch_pass, peg, lynch_reasons = peter_lynch_screen(pe, growth, peg_limit)
    intrinsic, graham_note = graham_intrinsic_value(eps, growth, bond_yield)

    mos_price = np.nan
    verdict = "N/A"
    mos_gap_pct = np.nan

    if np.isfinite(intrinsic):
        mos_price = intrinsic * (1 - mos_pct / 100.0)

    if np.isfinite(price) and np.isfinite(mos_price) and mos_price > 0:
        mos_gap_pct = (mos_price - price) / price * 100.0
        verdict = "Undervalued ✅" if price < mos_price else "Overvalued / No MOS ❌"

    return {
        "Ticker": tk,
        "Mkt Cap": mcap,
        "Current Price": price,
        "P/E": pe,
        "EPS (TTM)": eps,
        "Growth % (g)": growth,
        "Growth Source": growth_src,
        "PEG": peg,
        "Lynch Screen": "PASS" if lynch_pass else "FAIL",
        "Lynch Reasons": "; ".join(lynch_reasons) if lynch_reasons else "",
        "Graham Intrinsic": intrinsic,
        f"MOS Price ({mos_pct}%)": mos_price,
        "Verdict": verdict,
        "MOS Gap % (MOS - Price)": mos_gap_pct,
        "Graham Note": graham_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Premium Prediction", layout="wide")
st.title("🔮 AI Premium Prediction")

# Premium check (safe)
email = st.session_state.get("premium_email", None)
if not email:
    email = st.text_input("Confirm your email to access premium features", key="confirm_email_ai_page")

premium_users = st.session_state.get("premium_users", set())
if not email or email not in premium_users:
    st.error("Premium access required for this page. Please return to the main page and subscribe/verify.")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")
    st.stop()

st.success(f"Premium active for {email}")

tabs = st.tabs(["🤖 AI Forecast", "✨ Auto Value Picks (Lynch→Graham→MOS)", "📄 Screener.in CSV (Optional)"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: AI Forecast (kept)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🤖 AI Forecast")

    st.markdown("### Select Stock(s) / ETF(s)")
    search_options = load_search_options()

    selected_assets = st.multiselect(
        "🔍 Search & select stocks / ETFs (optional)",
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
            st.warning("No valid tickers could be resolved. Try RELIANCE.NS")

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

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend", str(analysis.get("Trend", "N/A")))
                    c2.metric("Volatility", str(analysis.get("Volatility", "N/A")))
                    c3.metric("Confidence", str(analysis.get("Confidence_Score", "N/A")))
                    st.info(f"Recommendation: **{analysis.get('Recommendation', 'N/A')}**")

                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(x=ai_df.index, y=ai_df["Predicted_Price"], name="AI Prediction"))
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
# TAB 2: Auto Value Picks (no manual screening by user)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("✨ Auto Value Picks (Lynch → Graham → 30% MOS)")

    st.caption(
        "This automatically scans a stock universe and returns a shortlist. "
        "No manual selection needed."
    )

    cA, cB, cC, cD = st.columns([1.0, 1.0, 1.0, 1.0])
    with cA:
        peg_limit = st.number_input("Lynch PEG limit", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with cB:
        mos_pct = st.number_input("Margin of Safety (%)", min_value=0, max_value=80, value=30, step=1)
    with cC:
        bond_yield = st.number_input(
            "Bond yield used in Graham formula (%)",
            min_value=0.5, max_value=20.0, value=8.0, step=0.1,
            help="Set an India-appropriate corporate/AAA yield you trust."
        )
    with cD:
        default_growth = st.number_input(
            "Default growth% (if missing)",
            min_value=0.0, max_value=30.0, value=12.0, step=0.5
        )

    # Universe config (not “manual stock picking”, just performance control)
    c1, c2 = st.columns([1, 1])
    with c1:
        universe_size = st.slider(
            "Universe size (auto-picked by market cap)", 100, 700, 300, 50,
            help="Larger universe = slower. Default 300 is a good balance."
        )
    with c2:
        top_n = st.slider("Show top N results", 10, 100, 30, 5)

    df_nse = load_nse_equity_df()
    if df_nse.empty:
        st.error("Could not load NSE equity list right now. Try again later or use Screener CSV tab.")
    else:
        # Step 1: pick top by market cap (requires marketCap fetch, but we do it fast-ish with threading)
        all_tickers = df_nse["TICKER"].tolist()

        if st.button("Run Auto Screening", type="primary"):
            with st.spinner("Step 1/3: Estimating market caps to select universe..."):
                # Fetch only market cap for many tickers (still uses get_info under the hood, cached)
                # We do threading to make it tolerable.
                mcap_rows = []
                max_workers = 24

                def mcap_only(tk):
                    f = fetch_fundamentals(tk)
                    return tk, f.get("market_cap", np.nan)

                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for tk in all_tickers[:2000]:  # NSE equity list can be >2000; cap to avoid insane runtime
                        futures.append(ex.submit(mcap_only, tk))

                    for fut in as_completed(futures):
                        tk, mc = fut.result()
                        if np.isfinite(mc) and mc > 0:
                            mcap_rows.append((tk, mc))

                if not mcap_rows:
                    st.error("Could not fetch market caps. Try smaller universe later or use Screener CSV tab.")
                else:
                    mcap_df = pd.DataFrame(mcap_rows, columns=["Ticker", "MktCap"]).sort_values("MktCap", ascending=False)
                    universe = mcap_df["Ticker"].head(universe_size).tolist()

                    with st.spinner("Step 2/3: Running Peter Lynch screen (P/E, growth, PEG)..."):
                        # Step 2+3 combined: compute value row for each ticker
                        rows = []
                        futures2 = []
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            for tk in universe:
                                futures2.append(ex.submit(
                                    compute_value_row, tk, peg_limit, mos_pct, bond_yield, default_growth
                                ))
                            for fut in as_completed(futures2):
                                rows.append(fut.result())

                        df = pd.DataFrame(rows)

                    with st.spinner("Step 3/3: Building shortlist..."):
                        # Shortlist rules: Lynch PASS + Undervalued with MOS
                        df["Mkt Cap"] = pd.to_numeric(df["Mkt Cap"], errors="coerce")
                        df_sorted = df.sort_values("Mkt Cap", ascending=False)

                        shortlist = df[
                            (df["Lynch Screen"] == "PASS") &
                            (df["Verdict"] == "Undervalued ✅")
                        ].copy()

                        shortlist = shortlist.sort_values("MOS Gap % (MOS - Price)", ascending=False)

                    st.success("Auto screening complete ✅")

                    st.markdown("### Shortlist (Lynch PASS + Undervalued with MOS)")
                    if shortlist.empty:
                        st.info("No stocks met both conditions with current inputs. Try raising PEG limit or growth default.")
                    else:
                        st.dataframe(
                            shortlist.head(top_n)[[
                                "Ticker", "Current Price", "P/E", "Growth % (g)", "PEG",
                                "Graham Intrinsic", f"MOS Price ({mos_pct}%)", "MOS Gap % (MOS - Price)",
                                "Growth Source"
                            ]],
                            use_container_width=True
                        )

                    st.markdown("### Full scan results (for transparency)")
                    st.dataframe(df_sorted, use_container_width=True)

                    st.markdown("### Notes")
                    st.write(
                        "- Yahoo data for Indian stocks can be incomplete; when growth is missing, the **default growth%** is used.\n"
                        "- If you want higher accuracy, use the Screener CSV tab (export → upload) and then apply Graham+MOS."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Screener.in CSV upload (compliant approach)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📄 Screener.in CSV (Optional) → Apply Graham + MOS")

    st.caption(
        "Screener.in doesn’t provide an API; the compliant way is: "
        "export your Screen results to CSV and upload here, then we apply Graham+MOS."
    )

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        mos_pct2 = st.number_input("Margin of Safety (%)", min_value=0, max_value=80, value=30, step=1, key="mos2")
    with cB:
        bond_yield2 = st.number_input(
            "Bond yield used in Graham formula (%)",
            min_value=0.5, max_value=20.0, value=8.0, step=0.1, key="y2"
        )
    with cC:
        default_growth2 = st.number_input("Default growth% (if missing)", 0.0, 30.0, 12.0, 0.5, key="g2")

    uploaded = st.file_uploader("Upload Screener.in exported CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df_up.head(20), use_container_width=True)

            # Try to find NSE symbol column
            # Screener exports typically include NSE Code / BSE Code / etc.
            possible_cols = [c for c in df_up.columns if str(c).strip().lower() in {"nse", "nse code", "nsecode", "nse_symbol", "symbol"}]
            col = possible_cols[0] if possible_cols else None

            if col is None:
                st.error("Could not find an NSE code column. Please ensure your CSV includes 'NSE' or 'NSE Code'.")
            else:
                tickers = []
                for x in df_up[col].dropna().astype(str):
                    sym = x.upper().strip()
                    if not sym.endswith(".NS"):
                        sym = sym + ".NS"
                    tickers.append(sym)

                tickers = list(dict.fromkeys(tickers))
                st.write(f"Found {len(tickers)} tickers.")

                if st.button("Apply Graham + MOS to Screener results", type="primary"):
                    with st.spinner("Calculating intrinsic values..."):
                        rows = []
                        max_workers = 24
                        futures = []
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            for tk in tickers[:800]:
                                futures.append(ex.submit(
                                    compute_value_row, tk, 999.0, mos_pct2, bond_yield2, default_growth2
                                ))
                            for fut in as_completed(futures):
                                rows.append(fut.result())

                        out = pd.DataFrame(rows)
                        out = out.sort_values("MOS Gap % (MOS - Price)", ascending=False)

                    st.markdown("### Results (sorted by best MOS gap)")
                    st.dataframe(out, use_container_width=True)

        except Exception as e:
            st.error(f"CSV read error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("← Back to Portfolio Analysis"):
    st.switch_page("Home.py")
