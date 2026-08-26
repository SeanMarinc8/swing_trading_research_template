import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("📈 Swing Trading & Crypto Dashboard")

# ---------------------------------------------------------
# Sidebar: Global Timeframe & Educational Guide
# ---------------------------------------------------------
st.sidebar.header("Settings")
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
timeframe = st.sidebar.selectbox("Period", timeframe_options, index=4)

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Indicator Cheat Sheet"):
  st.markdown("""
    * **RSI (Relative Strength Index):** Measures momentum (0–100).
      * **> 70:** Overbought (Potential pullback)
      * **< 30:** Oversold (Potential bounce)
    * **RVOL (Relative Volume):** Current volume vs. 20-period average.
      * **> 1.5x:** High trading activity / institutional pressure.
    * **ATR (Average True Range):** Volatility measure in dollars. Helps define stop-loss levels.
    * **Moving Averages (SMA):** Directional trends over 20 and 50 periods.
    """)


# Helper function to compute technical metrics
def compute_indicators(df):
  df["SMA_20"] = df["Close"].rolling(window=20).mean()
  df["SMA_50"] = df["Close"].rolling(window=50).mean()

  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))
  df["Overbought (70)"] = 70
  df["Oversold (30)"] = 30

  high_low = df["High"] - df["Low"]
  high_close = np.abs(df["High"] - df["Close"].shift())
  low_close = np.abs(df["Low"] - df["Close"].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  true_range = np.max(ranges, axis=1)
  df["ATR"] = true_range.rolling(14).mean()

  df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()
  return df


# Helper function to render metrics & charts
def render_dashboard(df, symbol_name, asset_type="Stock"):
  latest = df.iloc[-1]
  prev = df.iloc[-2]

  # Key Metrics
  col1, col2, col3, col4, col5 = st.columns(5)
  col1.metric(
      "Current Price",
      f"${latest['Close']:,.2f}",
      f"{(latest['Close']-prev['Close']):,.2f}",
  )
  col2.metric(
      "14-Period RSI",
      f"{latest['RSI']:.1f}" if pd.notnull(latest["RSI"]) else "N/A",
  )
  col3.metric(
      "RVOL", f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A"
  )
  col4.metric(
      "14-Period ATR",
      f"${latest['ATR']:,.2f}" if pd.notnull(latest["ATR"]) else "N/A",
  )
  col5.metric(
      "20-Period SMA",
      f"${latest['SMA_20']:,.2f}" if pd.notnull(latest["SMA_20"]) else "N/A",
  )

  # Interpretation Signals
  st.subheader("💡 Beginner Signals")
  rsi_val = latest["RSI"]
  rvol_val = latest["RVOL"]
  close_val = latest["Close"]
  sma20_val = latest["SMA_20"]

  signals = []
  if rsi_val >= 70:
    signals.append("⚠️ **Overbought (RSI > 70):** Price may be overextended.")
  elif rsi_val <= 30:
    signals.append(
        "🟢 **Oversold (RSI < 30):** Price in potential bounce territory."
    )
  else:
    signals.append("⚖️ **RSI Neutral:** Balanced momentum.")

  if close_val > sma20_val:
    signals.append("📈 **Short-Term Uptrend:** Price above 20-period average.")
  else:
    signals.append("📉 **Short-Term Downtrend:** Price below 20-period average.")

  if rvol_val >= 1.5:
    signals.append("🔥 **Elevated Volume:** High activity detected.")

  st.info(" | ".join(signals))

  # Charts
  st.subheader(f"Price Action ({symbol_name})")
  st.line_chart(df[["Close", "SMA_20", "SMA_50"]])

  st.subheader("RSI Momentum & Thresholds")
  st.line_chart(df[["RSI", "Overbought (70)", "Oversold (30)"]])


# ---------------------------------------------------------
# App Tabs: Stocks vs Crypto
# ---------------------------------------------------------
tab_stocks, tab_crypto = st.tabs(["📊 Stocks & ETFs", "🪙 Cryptocurrency"])

# TAB 1: STOCKS
with tab_stocks:
  stock_ticker = st.text_input(
      "Stock Ticker", value="VOO", key="stock_input"
  ).upper()
  if stock_ticker:
    data = yf.Ticker(stock_ticker).history(period=timeframe)
    if not data.empty:
      df_stock = compute_indicators(data)
      render_dashboard(df_stock, stock_ticker, asset_type="Stock")
    else:
      st.error(f"Could not load data for stock ticker: {stock_ticker}")

# TAB 2: CRYPTO
with tab_crypto:
  st.caption(
      "Note: Crypto trades 24/7. Tickers require currency pairs (e.g., BTC-USD,"
      " ETH-USD, SOL-USD)."
  )

  # Quick selection selectbox + custom text input option
  preset_crypto = st.selectbox(
      "Select Popular Crypto or Custom",
      ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "Custom Input"],
  )

  if preset_crypto == "Custom Input":
    crypto_ticker = st.text_input(
        "Enter Crypto Pair (e.g., ADA-USD)", value="ADA-USD", key="crypto_input"
    ).upper()
  else:
    crypto_ticker = preset_crypto

  if crypto_ticker:
    crypto_data = yf.Ticker(crypto_ticker).history(period=timeframe)
    if not crypto_data.empty:
      df_crypto = compute_indicators(crypto_data)
      render_dashboard(df_crypto, crypto_ticker, asset_type="Crypto")
    else:
      st.error(
          f"Could not load crypto data for: {crypto_ticker}. Make sure to include"
          " '-USD' at the end."
      )
