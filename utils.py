import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker, period="1y"):
    """
    Fetches stock data for a given ticker.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def simple_prediction(data, days=30):
    """
    Simple prediction using Moving Average.
    """
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    last_ma = data['MA_50'].iloc[-1]
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1)[1:]
    
    # Simple linear projection (mock AI)
    predictions = [last_ma * (1 + np.random.normal(0, 0.01)) for _ in range(days)]
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': predictions}).set_index('Date')

def advanced_ai_prediction(ticker, days=30):
    """
    Simulates a premium AI agent prediction.
    In a real app, this would use a more complex model or API.
    """
    # Mocking a "thinking" process and advanced result
    data = get_stock_data(ticker, period="2y")
    
    # Mock advanced analysis
    volatility = data['Close'].std()
    trend = "Bullish" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "Bearish"
    
    # Generate 'smarter' predictions with confidence intervals
    last_price = data['Close'].iloc[-1]
    future_dates = pd.date_range(start=data.index[-1], periods=days + 1)[1:]
    
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    current_price = last_price
    for _ in range(days):
        change = np.random.normal(0, volatility * 0.05) # Reduced volatility for prediction
        current_price += change
        predictions.append(current_price)
        lower_bounds.append(current_price - (volatility * 0.1))
        upper_bounds.append(current_price + (volatility * 0.1))
        
    df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions,
        'Lower_Bound': lower_bounds,
        'Upper_Bound': upper_bounds
    }).set_index('Date')
    
    analysis = {
        "Trend": trend,
        "Volatility": f"{volatility:.2f}",
        "Recommendation": "Buy" if trend == "Bullish" else "Sell",
        "Confidence_Score": "87%" # Mock score
    }
    
    return df, analysis
