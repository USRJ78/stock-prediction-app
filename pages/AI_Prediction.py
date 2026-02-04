import streamlit as st
import pandas as pd
import yfinance as yf
import time

st.title("Portfolio Construction Dashboard")
st.markdown("""
**Live Value + Growth Screening**  
1. Click "Start Live Screening" to fetch current data  
2. First filter: Peter Lynch style (low PEG < 1 preferred)  
3. Then apply Benjamin Graham intrinsic value + 30% margin of safety  
Data from yfinance (live prices) + recent low-PEG candidates. Always cross-check on Screener.in.
""")

# Candidate stocks from recent low-PEG screens (Feb 2026 context - high growth + reasonable valuation)
# You can expand this list over time
candidates = [
    "PREMIER.NS", "WELSPUNCORP.NS", "ZENT.NS", "NATCOPHARM.NS", "INSOL.NS",
    "GANESHHOUC.NS", "SHILCTRN.NS", "TBO.NS", "KPITTECH.NS", "SHAKTIPUMP.NS",
    "SCHNEIDER.NS", "BILLIONBRAINS.NS", "CEATLTD.NS", "JUPITER.NS", "DOMS.NS",
    # Add more tickers as you discover low-PEG ones
]

if st.button("Start Live Screening", type="primary"):
    with st.spinner("Fetching live data and running screening... (may take 10-30 seconds)"):
        results = []
        bond_yield = 7.6  # AAA corporate India ~7.6% (early 2026)

        for ticker in candidates:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
                if current_price is None:
                    continue

                eps = info.get('trailingEps', info.get('forwardEps', None))
                if eps is None or eps <= 0:
                    continue

                growth_pct = info.get('earningsGrowth', 0) * 100
                if growth_pct <= 0:
                    growth_pct = 10  # fallback reasonable growth

                div_yield_pct = (info.get('dividendYield', 0) or 0) * 100

                current_pe = info.get('trailingPE', info.get('forwardPE', None)) or (current_price / eps)

                # Peter Lynch
                lynch_fair_pe = growth_pct + div_yield_pct
                lynch_fv = eps * lynch_fair_pe
                peg = current_pe / growth_pct if growth_pct > 0 else 999

                # Benjamin Graham
                graham_base = 8.5 + 2 * growth_pct
                graham_v = (eps * graham_base * 4.4) / bond_yield
                graham_mos = graham_v * 0.70  # 30% margin of safety

                # Verdicts
                lynch_verdict = "Undervalued" if current_price < lynch_fv else "Fair" if abs(current_price - lynch_fv)/lynch_fv < 0.15 else "Overvalued"
                graham_verdict = "Undervalued (MoS)" if current_price < graham_mos else "Fair" if current_price < graham_v else "Overvalued"

                results.append({
                    "Ticker": ticker,
                    "Company": info.get('shortName', ticker),
                    "Current Price": round(current_price, 1),
                    "PEG": round(peg, 2) if peg < 999 else "-",
                    "Lynch Fair Value": round(lynch_fv, 0),
                    "Lynch Verdict": lynch_verdict,
                    "Graham Intrinsic": round(graham_v, 0),
                    "Graham +30% MoS": round(graham_mos, 0),
                    "Graham Verdict": graham_verdict
                })

                time.sleep(0.5)  # polite delay to avoid rate limits

            except Exception as e:
                continue  # skip failed tickers

        if results:
            df = pd.DataFrame(results)
            st.subheader("Live Screening Results")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Current Price": st.column_config.NumberColumn("Price (₹)"),
                    "PEG": st.column_config.NumberColumn("PEG", format="%.2f"),
                    "Lynch Fair Value": st.column_config.NumberColumn("Lynch FV (₹)", format="%.0f"),
                    "Graham Intrinsic": st.column_config.NumberColumn("Graham IV (₹)", format="%.0f"),
                    "Graham +30% MoS": st.column_config.NumberColumn("Graham MoS (₹)", format="%.0f"),
                }
            )

            st.success(f"Screened {len(results)} stocks with live data.")
            st.info("Undervalued stocks (especially those undervalued on both) are top candidates. Verify latest data on Screener.in.")
        else:
            st.error("No valid data fetched. Check internet or try again later.")
else:
    st.info("Click 'Start Live Screening' to fetch current market data and run the Peter Lynch → Benjamin Graham process.")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to AI Prediction"):
        st.switch_page("pages/AI_Prediction.py")
with col2:
    if st.button("← Back to Portfolio Home"):
        st.switch_page("Home.py")
