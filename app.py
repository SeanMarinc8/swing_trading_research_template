import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("📈 Swing Trading Dashboard")

# Sidebar Controls
ticker = st.sidebar.text_input("Enter Ticker", value="VOO").upper()
timeframe_options = [
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]
timeframe = st.sidebar.selectbox("Period", timeframe_options, index=5)

if ticker:
  stock = yf.Ticker(ticker)
  df = stock.history(period=timeframe)

  if not df.empty:
    # 1. Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # 2. RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. ATR Calculation
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # 4. Relative Volume (RVOL)
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Metrics Display
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Current Price",
        f"${latest['Close']:.2f}",
        f"{(latest['Close']-prev['Close']):.2f}",
    )
    col2.metric(
        "14-Day RSI",
        f"{latest['RSI']:.1f}" if pd.notnull(latest["RSI"]) else "N/A",
    )
    col3.metric(
        "RVOL",
        f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A",
    )
    col4.metric(
        "14-Day ATR",
        f"${latest['ATR']:.2f}" if pd.notnull(latest["ATR"]) else "N/A",
    )
    col5.metric(
        "20-Day SMA",
        f"${latest['SMA_20']:.2f}" if pd.notnull(latest["SMA_20"]) else "N/A",
    )

    # Charts
    st.subheader(f"Price Action ({ticker})")
    st.line_chart(df[["Close", "SMA_20", "SMA_50"]])

    st.subheader("RSI Momentum")
    st.line_chart(df["RSI"])

  else:
    st.error("Invalid Ticker or No Data Found")
