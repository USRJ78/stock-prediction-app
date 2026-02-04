import streamlit as st
import pandas as pd
import yfinance as yf
import time

st.title("Live Portfolio Value Screening")

st.markdown("""
**Live GARP + Value Dashboard**  
- Click **Start Screening** to fetch current prices, EPS, growth, etc.  
- First filter: Peter Lynch style (PEG < 1.2 preferred)  
- Then apply Benjamin Graham + **30% margin of safety**  
- Results are live — small changes between runs are normal (market moves).  
- Based on recent low-PEG candidates (expand list anytime).
""")

# Reduced candidate list for speed (10 stocks — all recent low-PEG examples)
candidates = [
    "PREMIER.NS", "WELSPUNCORP.NS", "ZENT.NS", "NATCOPHARM.NS", "INSOL.NS",
    "GANESHHOUC.NS", "SHILCTRN.NS", "TBO.NS", "KPITTECH.NS", "SHAKTIPUMP.NS"
]

@st.cache_data(ttl=300)  # cache for 5 min to speed repeated runs
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'eps': info.get('trailingEps') or info.get('forwardEps'),
            'growth_pct': (info.get('earningsGrowth') or 0) * 100,
            'div_yield_pct': (info.get('dividendYield') or 0) * 100,
            'pe': info.get('trailingPE') or info.get('forwardPE'),
            'name': info.get('shortName', ticker)
        }
    except:
        return None

if st.button("Start Live Screening", type="primary", use_container_width=True):
    with st.spinner("Screening live data..."):
        results = []
        bond_yield = 7.6

        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(candidates):
            status.text(f"Checking {ticker} ({i+1}/{len(candidates)})")
            data = get_stock_data(ticker)
            if not data or not data['current_price'] or not data['eps'] or data['eps'] <= 0:
                continue

            growth_pct = data['growth_pct'] if data['growth_pct'] > 0 else 10
            current_pe = data['pe'] or (data['current_price'] / data['eps'])
            peg = current_pe / growth_pct if growth_pct > 0 else 999

            # Lynch filter first
            if peg > 1.2:
                continue  # skip if not promising

            lynch_fair_pe = growth_pct + data['div_yield_pct']
            lynch_fv = data['eps'] * lynch_fair_pe

            # Graham
            graham_base = 8.5 + 2 * growth_pct
            graham_v = (data['eps'] * graham_base * 4.4) / bond_yield
            graham_mos = graham_v * 0.70

            lynch_verdict = "Undervalued" if data['current_price'] < lynch_fv else "Fair" if abs(data['current_price'] - lynch_fv)/lynch_fv < 0.15 else "Overvalued"
            graham_verdict = "Undervalued (MoS)" if data['current_price'] < graham_mos else "Fair" if data['current_price'] < graham_v else "Overvalued"

            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Company": data['name'],
                "Price": round(data['current_price'], 1),
                "PEG": round(peg, 2),
                "Lynch FV": round(lynch_fv, 0),
                "Lynch Verdict": lynch_verdict,
                "Graham IV": round(graham_v, 0),
                "Graham MoS": round(graham_mos, 0),
                "Graham Verdict": graham_verdict
            })

            progress.progress((i + 1) / len(candidates))
            time.sleep(0.3)  # minimal delay

        progress.empty()
        status.empty()

        if results:
            df = pd.DataFrame(results)
            st.subheader(f"Live Results ({len(results)} stocks passed)")
            st.dataframe(
                df.sort_values("Graham MoS", ascending=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn("Price (₹)"),
                    "PEG": st.column_config.NumberColumn("PEG", format="%.2f"),
                    "Lynch FV": st.column_config.NumberColumn("Lynch FV (₹)", format="%.0f"),
                    "Graham IV": st.column_config.NumberColumn("Graham IV (₹)", format="%.0f"),
                    "Graham MoS": st.column_config.NumberColumn("Buy Below (MoS ₹)", format="%.0f"),
                }
            )
            st.success("Screening done! Results are live — refresh anytime for latest data.")
        else:
            st.warning("No stocks passed filters this time. Market may have shifted — try again later.")
else:
    st.info("Press 'Start Live Screening' to run the real-time process.")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to AI Prediction"):
        st.switch_page("pages/AI_Prediction.py")
with col2:
    if st.button("← Back to Home"):
        st.switch_page("Home.py")
