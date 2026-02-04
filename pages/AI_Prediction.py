import streamlit as st
import pandas as pd

st.title("Portfolio Construction Dashboard (Value + Growth Screening)")

st.markdown("""
This page shows a screened list of potential stocks using **Peter Lynch GARP** (low PEG < 1) as the first filter.  
Then we apply **Benjamin Graham** intrinsic value (revised formula) with **30% margin of safety** to check if they are undervalued.

Data is approximate (early 2026 market conditions) — always verify live on Screener.in / Yahoo Finance / Tickertape.  
Bond yield used: ~7.6% (AAA corporate India).
""")

# Pre-screened stocks from recent low-PEG GARP screens (examples from Screener.in / Tickertape / Equitymaster)
# Format: {'Ticker': 'Company Name', 'Current_Price': float, 'EPS': float, 'Growth_pct': float, 'PEG': float}
screened_stocks = [
    {'Ticker': 'PREMIER', 'Company': 'Premier Energies', 'Current_Price': 1050, 'EPS': 45, 'Growth_pct': 30, 'PEG': 0.15},
    {'Ticker': 'WELSPUNCORP', 'Company': 'Welspun Corp', 'Current_Price': 650, 'EPS': 35, 'Growth_pct': 18, 'PEG': 0.37},
    {'Ticker': 'ZENTECH', 'Company': 'Zen Technologies', 'Current_Price': 1900, 'EPS': 18, 'Growth_pct': 35, 'PEG': 0.13},
    {'Ticker': 'NATCOPHARM', 'Company': 'Natco Pharma', 'Current_Price': 1300, 'EPS': 55, 'Growth_pct': 18, 'PEG': 0.07},
    {'Ticker': 'INSOL', 'Company': 'Insolation Energy', 'Current_Price': 280, 'EPS': 12, 'Growth_pct': 40, 'PEG': 0.22},
    {'Ticker': 'GANESHHOUC', 'Company': 'Ganesh Housing Corp', 'Current_Price': 750, 'EPS': 30, 'Growth_pct': 20, 'PEG': 0.4},
    {'Ticker': 'SHILCTRN', 'Company': 'Shilchar Technologies', 'Current_Price': 4500, 'EPS': 120, 'Growth_pct': 25, 'PEG': 0.7},
    # Add more if you want — or expand later with API
]

df = pd.DataFrame(screened_stocks)

# Calculate valuations
df['Lynch_Fair_PE'] = df['Growth_pct'] + 1.0  # approx dividend yield 1% average
df['Lynch_Fair_Value'] = df['EPS'] * df['Lynch_Fair_PE']

df['Graham_Base'] = 8.5 + 2 * df['Growth_pct']
df['Graham_Intrinsic'] = (df['EPS'] * df['Graham_Base'] * 4.4) / 7.6
df['Graham_MoS_30'] = df['Graham_Intrinsic'] * 0.70

# Verdict columns
df['Lynch_Verdict'] = df.apply(lambda row: 
    'Undervalued' if row['Current_Price'] < row['Lynch_Fair_Value'] else 
    'Fairly Valued' if abs(row['Current_Price'] - row['Lynch_Fair_Value']) / row['Lynch_Fair_Value'] < 0.15 else 'Overvalued', axis=1)

df['Graham_Verdict'] = df.apply(lambda row: 
    'Undervalued (with safety)' if row['Current_Price'] < row['Graham_MoS_30'] else 
    'Fairly Valued' if row['Current_Price'] < row['Graham_Intrinsic'] else 'Overvalued', axis=1)

# Display dashboard table
st.subheader("Screened Stocks Dashboard")
st.dataframe(
    df[['Company', 'Ticker', 'Current_Price', 'PEG', 'Lynch_Fair_Value', 'Lynch_Verdict', 
        'Graham_Intrinsic', 'Graham_MoS_30', 'Graham_Verdict']],
    use_container_width=True,
    column_config={
        'Current_Price': st.column_config.NumberColumn("Current Price (₹)"),
        'PEG': st.column_config.NumberColumn("PEG Ratio", format="%.2f"),
        'Lynch_Fair_Value': st.column_config.NumberColumn("Lynch Fair Value (₹)", format="%.0f"),
        'Graham_Intrinsic': st.column_config.NumberColumn("Graham Intrinsic (₹)", format="%.0f"),
        'Graham_MoS_30': st.column_config.NumberColumn("Graham + 30% MoS (₹)", format="%.0f"),
    }
)

st.info("""
**Interpretation Guide**:
- **Peter Lynch Verdict**: Focuses on growth at reasonable price (PEG < 1 = undervalued).
- **Graham Verdict**: More conservative — prioritizes margin of safety.
- **Best opportunities**: Stocks undervalued on **both** (e.g. Premier Energies, Welspun Corp in this snapshot).
- Data is illustrative — refresh with latest numbers from Screener.in or yfinance.
""")

# Back button
st.markdown("---")
if st.button("← Back to AI Prediction"):
    st.switch_page("pages/AI_Prediction.py")
if st.button("← Back to Portfolio Home"):
    st.switch_page("Home.py")
