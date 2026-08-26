import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pro Swing & Day Trading Scanner", layout="wide")
st.title("📊 Institutional Grade Stock Dashboard")

# ---------------------------------------------------------
# Sidebar Inputs & Glossary
# ---------------------------------------------------------
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
timeframe_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
timeframe = st.sidebar.selectbox("Period", timeframe_options, index=3)

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Pro Metrics Guide"):
  st.markdown("""
    * **VWAP:** Volume-Weighted Average Price. Price above VWAP = Bullish institutional bias.
    * **Bollinger %B:** Measures price location relative to bands (>1.0 = above upper band, <0.0 = below lower band).
    * **BB Squeeze:** Extremely narrow bands indicating an imminent volatile breakout.
    * **MACD Hist:** Shows momentum acceleration/deceleration before trend flips.
    """)

if ticker:
  df = yf.Ticker(ticker).history(period=timeframe)

  if not df.empty:
    # 1. Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # 2. VWAP (Volume-Weighted Average Price)
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    # 3. RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 4. Bollinger Bands (%B & Bandwidth Squeeze)
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (std_20 * 2)
    df["BB_Lower"] = df["SMA_20"] - (std_20 * 2)
    df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]

    # 5. MACD & MACD Histogram
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 6. ATR & RVOL
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["ATR"] = np.max(ranges, axis=1).rolling(14).mean()
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Metrics Display
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${latest['Close']:.2f}", f"{(latest['Close']-prev['Close']):.2f}")
    col2.metric("VWAP", f"${latest['VWAP']:.2f}" if pd.notnull(latest["VWAP"]) else "N/A")
    col3.metric("Bollinger %B", f"{latest['BB_Percent']:.2f}" if pd.notnull(latest["BB_Percent"]) else "N/A")
    col4.metric("RVOL", f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A")
    col5.metric("14-Day ATR", f"${latest['ATR']:.2f}" if pd.notnull(latest["ATR"]) else "N/A")

    # Professional Trade Setup Diagnostics
    st.subheader("🎯 Institutional Signal Panel")
    insights = []

    # VWAP Bias
    if latest["Close"] > latest["VWAP"]:
      insights.append("🟢 **Bullish Bias:** Price trading above VWAP (Buyers in control).")
    else:
      insights.append("🔴 **Bearish Bias:** Price trading below VWAP (Sellers in control).")

    # Bollinger Band Squeeze Check
    min_width_20 = df["BB_Width"].rolling(20).min().iloc[-1]
    if latest["BB_Width"] <= min_width_20:
      insights.append("⚡ **Volatility Squeeze:** Bollinger Bands narrowed to 20-period low. Expect high expansion move.")

    # MACD Momentum Shift
    if latest["MACD_Hist"] > 0 and prev["MACD_Hist"] <= 0:
      insights.append("🚀 **MACD Momentum Flip:** Histogram turned positive (Bullish momentum expansion).")
    elif latest["MACD_Hist"] < 0 and prev["MACD_Hist"] >= 0:
      insights.append("⚠️ **MACD Momentum Flip:** Histogram turned negative (Bearish momentum expansion).")

    st.info(" | ".join(insights))

    # Charts
    st.subheader(f"Price Action vs. VWAP & Bollinger Bands ({ticker})")
    st.line_chart(df[["Close", "VWAP", "BB_Upper", "BB_Lower"]])

    st.subheader("MACD Histogram & Momentum")
    st.bar_chart(df["MACD_Hist"])

  else:
    st.error("Invalid Ticker or No Data Available")
