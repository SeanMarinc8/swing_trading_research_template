import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("📈 Swing Trading Dashboard")

# 1. Sets VOO as the default ticker input
ticker = st.sidebar.text_input("Enter Ticker", value="VOO").upper()

# 2. Offers expanded yfinance timeframe options
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
timeframe = st.sidebar.selectbox(
    "Period", timeframe_options, index=5
)  # Defaults to '1y'

if ticker:
  stock = yf.Ticker(ticker)
  df = stock.history(period=timeframe)

  if not df.empty:
    # Calculations
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Display Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Current Price",
        f"${latest['Close']:.2f}",
        f"{(latest['Close']-prev['Close']):.2f}",
    )
    col2.metric("14-Day RSI", f"{latest['RSI']:.1f}")
    col3.metric("RVOL", f"{latest['RVOL']:.2f}x")
    col4.metric("14-Day ATR", f"${latest['ATR']:.2f}")
    col5.metric("20-Day SMA", f"${latest['SMA_20']:.2f}")

    # Charts
    st.subheader(f"Price Action ({ticker})")
    st.line_chart(df[["Close", "SMA_20", "SMA_50"]])

    st.subheader("RSI Momentum")
    st.line_chart(df["RSI"])
  else:
    st.error("Invalid Ticker")
