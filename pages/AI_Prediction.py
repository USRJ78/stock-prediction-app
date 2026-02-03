import streamlit as st
import yfinance as yf
from utils import advanced_ai_prediction

st.title("🔮 AI Premium Prediction")

# Basic premium check using session state
email = st.session_state.get("premium_email", None)
if not email or email not in st.session_state.premium_users:
    st.error("Premium access required for this page.")
    if st.button("← Back to Portfolio"):
        st.switch_page("Home.py")  # change if your main file has a different name
    st.stop()

st.success(f"Premium active for {email}")

chosen_ticker = st.text_input("Stock Ticker", "RELIANCE.NS").upper()

horizon_map = {"1W": 5, "1M": 21, "3M": 63, "1Y": 252}
horizon_label = st.selectbox("Horizon", list(horizon_map.keys()), index=1)
horizon_days = horizon_map[horizon_label]

if st.button("Run AI Prediction"):
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

st.markdown("---")
if st.button("← Back to Portfolio"):
    st.switch_page("Home.py")  # change if needed
