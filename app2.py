import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("📈 Swing Trading Dashboard")

# ---------------------------------------------------------
# Sidebar: Controls & Beginner Glossary
# ---------------------------------------------------------
st.sidebar.header("Controls")
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

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Indicator Cheat Sheet for Beginners"):
  st.markdown("""
    * **RSI (Relative Strength Index):** Measures momentum on a 0–100 scale.
      * **> 70:** Overbought (due for a pullback)
      * **< 30:** Oversold (potential buying opportunity)
    * **RVOL (Relative Volume):** Compares today's volume to the 20-day average. 
      * **> 1.5x:** Institutional interest / high movement.
    * **ATR (Average True Range):** Average daily price swing in dollars. Helps set realistic stop-losses.
    * **20 & 50 SMAs:** Simple Moving Averages showing short- and medium-term trend direction.
    """)

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

    # Add RSI Threshold lines for chart reference
    df["Overbought (70)"] = 70
    df["Oversold (30)"] = 30

    # 3. ATR Calculation
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # 4. Relative Volume (RVOL)
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Latest Values
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Metrics Display with Native Tooltips
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Current Price",
        f"${latest['Close']:.2f}",
        f"{(latest['Close']-prev['Close']):.2f}",
        help="Latest closing price and 1-day change.",
    )
    col2.metric(
        "14-Day RSI",
        f"{latest['RSI']:.1f}" if pd.notnull(latest["RSI"]) else "N/A",
        help="Momentum indicator (30 = Oversold, 70 = Overbought)",
    )
    col3.metric(
        "RVOL",
        f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A",
        help="Volume relative to 20-day average. > 1.0 means higher activity than normal.",
    )
    col4.metric(
        "14-Day ATR",
        f"${latest['ATR']:.2f}" if pd.notnull(latest["ATR"]) else "N/A",
        help="Expected daily dollar movement range based on recent volatility.",
    )
    col5.metric(
        "20-Day SMA",
        f"${latest['SMA_20']:.2f}" if pd.notnull(latest["SMA_20"]) else "N/A",
        help="Average price over the last 20 trading days (Short-term trend line).",
    )

    # ---------------------------------------------------------
    # Plain-English Takeaway Banner
    # ---------------------------------------------------------
    st.subheader("💡 Beginner Interpretation")
    rsi_val = latest["RSI"]
    rvol_val = latest["RVOL"]
    close_val = latest["Close"]
    sma20_val = latest["SMA_20"]

    status_signals = []

    # RSI Interpretation
    if rsi_val >= 70:
      status_signals.append(
          "⚠️ **RSI is Overbought (>70):** The stock may be overextended short-term."
      )
    elif rsi_val <= 30:
      status_signals.append(
          "🟢 **RSI is Oversold (<30):** Potential reversal/bounce territory."
      )
    else:
      status_signals.append("⚖️ **RSI is Neutral:** Momentum is balanced.")

    # Trend Interpretation
    if close_val > sma20_val:
      status_signals.append(
          "📈 **Short-Term Uptrend:** Price is trading above the 20-day average."
      )
    else:
      status_signals.append(
          "📉 **Short-Term Downtrend:** Price is trading below the 20-day average."
      )

    # Volume Interpretation
    if rvol_val >= 1.5:
      status_signals.append(
          "🔥 **High Relative Volume:** Strong trading activity relative to recent history."
      )

    st.info(" | ".join(status_signals))

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------
    st.subheader(f"Price Action & Moving Averages ({ticker})")
    st.line_chart(df[["Close", "SMA_20", "SMA_50"]])

    st.subheader("RSI Momentum & Bounds")
    # Displays RSI alongside fixed reference lines at 30 and 70
    st.line_chart(df[["RSI", "Overbought (70)", "Oversold (30)"]])

  else:
    st.error("Invalid Ticker or No Data Found")
