import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Buy / Sell Signals")

# -----------------------------
# FILE INPUT
# -----------------------------
stock_name = st.text_input("Enter stock name (csv file name without extension)")
data_points = st.slider("Number of datapoints",50,500,200)

if stock_name:
    path=f"./data/{stock_name}.csv"
    df=pd.read_csv(path)

    # FIX DATE ISSUE
    if "Price" in df.columns:
        df=df.rename(columns={"Price":"Date"})

    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df=df.iloc[2:]

    numeric_cols=['Adj Close','Close','High','Low','Open','Volume']
    for c in numeric_cols:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce")

    df=df.set_index("Date")

    # -----------------------------
    # TIMEFRAME OPTIONS
    # -----------------------------
    timeframe=st.selectbox(
        "Select timeframe",
        ["1D","4H","1W","1M"]
    )

    if timeframe=="4H":
        df=df.resample("4H").agg({
            "Open":"first",
            "High":"max",
            "Low":"min",
            "Close":"last",
            "Volume":"sum"
        }).dropna()

    if timeframe=="1W":
        df=df.resample("W").last()

    if timeframe=="1M":
        df=df.resample("M").last()

    df=df.tail(data_points)

    # -----------------------------
    # INDICATORS
    # -----------------------------
    df["20_SMA"]=df["Close"].rolling(20).mean()
    df["50_SMA"]=df["Close"].rolling(50).mean()

    df["ATR"]=(
        pd.concat([
            df["High"]-df["Low"],
            abs(df["High"]-df["Close"].shift()),
            abs(df["Low"]-df["Close"].shift())
        ],axis=1)
    ).max(axis=1)

    df["Volatility"]=df["Close"].pct_change().rolling(10).std()

    # -----------------------------
    # SIGNAL TYPE
    # -----------------------------
    signal_type=st.selectbox(
        "Signal Type",
        ["Golden Cross","UT Bot Alerts"]
    )

    # -----------------------------
    # GOLDEN CROSS
    # -----------------------------
    if signal_type=="Golden Cross":

        df["Signal"]=0
        df.loc[df["20_SMA"]>df["50_SMA"],"Signal"]=1
        df["Position"]=df["Signal"].diff()

        buy=df[df["Position"]==1]
        sell=df[df["Position"]==-1]

    # -----------------------------
    # UT BOT ALERTS (TV STYLE)
    # -----------------------------
    else:

        key=2
        atr_period=1

        df["ATR"]=df["ATR"].rolling(atr_period).mean()

        df["nLoss"]=key*df["ATR"]

        df["xATRTrailingStop"]=0.0

        for i in range(1,len(df)):

            prev_stop=df["xATRTrailingStop"].iloc[i-1]
            prev_close=df["Close"].iloc[i-1]
            close=df["Close"].iloc[i]

            if close>prev_stop and prev_close>prev_stop:
                df.loc[df.index[i],"xATRTrailingStop"]=max(prev_stop,close-df["nLoss"].iloc[i])

            elif close<prev_stop and prev_close<prev_stop:
                df.loc[df.index[i],"xATRTrailingStop"]=min(prev_stop,close+df["nLoss"].iloc[i])

            elif close>prev_stop:
                df.loc[df.index[i],"xATRTrailingStop"]=close-df["nLoss"].iloc[i]

            else:
                df.loc[df.index[i],"xATRTrailingStop"]=close+df["nLoss"].iloc[i]

        df["Position"]=0

        df.loc[
            (df["Close"]>df["xATRTrailingStop"]) &
            (df["Close"].shift()<=df["xATRTrailingStop"].shift()),
            "Position"
        ]=1

        df.loc[
            (df["Close"]<df["xATRTrailingStop"]) &
            (df["Close"].shift()>=df["xATRTrailingStop"].shift()),
            "Position"
        ]=-1

        buy=df[df["Position"]==1]
        sell=df[df["Position"]==-1]

    # -----------------------------
    # CHART OPTIONS
    # -----------------------------
    chart_type=st.selectbox(
        "Chart Type",
        ["Line","Candlestick"]
    )

    chart_dim=st.selectbox(
        "Chart Dimension",
        ["2D","3D"]
    )

    # -----------------------------
    # THIRD AXIS OPTION
    # -----------------------------
    z_axis=st.selectbox(
        "3rd Axis (for 3D)",
        ["Volume","Volatility","ATR","SMA Spread"]
    )

    if z_axis=="Volume":
        df["Z"]=df["Volume"]

    if z_axis=="Volatility":
        df["Z"]=df["Volatility"]

    if z_axis=="ATR":
        df["Z"]=df["ATR"]

    if z_axis=="SMA Spread":
        df["Z"]=df["20_SMA"]-df["50_SMA"]

    # -----------------------------
    # RESET INDEX (FIXES KEYERROR)
    # -----------------------------
    df=df.reset_index()

    buy=buy.reset_index()
    sell=sell.reset_index()

    # -----------------------------
    # 2D CHART
    # -----------------------------
    if chart_dim=="2D":

        fig=go.Figure()

        if chart_type=="Line":
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="Close"
            ))

        else:
            fig.add_trace(go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["20_SMA"],
            name="20 SMA"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["50_SMA"],
            name="50 SMA"
        ))

        fig.add_trace(go.Scatter(
            x=buy["Date"],
            y=buy["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up",size=12),
            name="Buy"
        ))

        fig.add_trace(go.Scatter(
            x=sell["Date"],
            y=sell["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down",size=12),
            name="Sell"
        ))

        st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # 3D CHART
    # -----------------------------
    else:

        fig=go.Figure()

        fig.add_trace(go.Scatter3d(
            x=df["Date"],
            y=df["Close"],
            z=df["Z"],
            mode="lines",
            name="Price"
        ))

        fig.add_trace(go.Scatter3d(
            x=buy["Date"],
            y=buy["Close"],
            z=buy["Z"],
            mode="markers",
            marker=dict(size=6,symbol="diamond"),
            name="Buy"
        ))

        fig.add_trace(go.Scatter3d(
            x=sell["Date"],
            y=sell["Close"],
            z=sell["Z"],
            mode="markers",
            marker=dict(size=6,symbol="diamond"),
            name="Sell"
        ))

        st.plotly_chart(fig,use_container_width=True)
