from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Institutional Swing & Crypto Scanner", layout="wide"
)
st.title("📈 Swing & Crypto Dashboard with Predictive Engine")

# ---------------------------------------------------------
# Sidebar Settings & Reference
# ---------------------------------------------------------
st.sidebar.header("Global Controls")
timeframe_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
timeframe = st.sidebar.selectbox("Analysis Horizon", timeframe_options, index=4)

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Technical Glossary"):
  st.markdown("""
    * **VWAP:** Volume-Weighted Average Price. Defines institutional benchmark value.
    * **Bollinger %B:** Measures price position relative to volatility bands (0.0 = Lower Band, 1.0 = Upper Band).
    * **MACD Hist:** Measures acceleration/deceleration of price momentum.
    * **ATR (Average True Range):** Daily volatility expected movement in dollars.
    * **RVOL:** Current volume vs 20-period average (>1.5x indicates heavy trading activity).
    """)


# ---------------------------------------------------------
# Computation Engine
# ---------------------------------------------------------
def compute_all_indicators(df):
  df = df.copy()

  # Moving Averages
  df["SMA_20"] = df["Close"].rolling(window=20).mean()
  df["SMA_50"] = df["Close"].rolling(window=50).mean()

  # Volume-Weighted Average Price (VWAP)
  tp = (df["High"] + df["Low"] + df["Close"]) / 3
  df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

  # RSI Calculation
  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))
  df["Overbought (70)"] = 70
  df["Oversold (30)"] = 30

  # Bollinger Bands (%B & Width)
  std_20 = df["Close"].rolling(window=20).std()
  df["BB_Upper"] = df["SMA_20"] + (std_20 * 2)
  df["BB_Lower"] = df["SMA_20"] - (std_20 * 2)
  df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (
      df["BB_Upper"] - df["BB_Lower"]
  )
  df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]

  # MACD Histogram
  ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
  ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
  df["MACD"] = ema_12 - ema_26
  df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

  # ATR & RVOL
  high_low = df["High"] - df["Low"]
  high_close = np.abs(df["High"] - df["Close"].shift())
  low_close = np.abs(df["Low"] - df["Close"].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  df["ATR"] = np.max(ranges, axis=1).rolling(14).mean()
  df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

  return df


def generate_predictive_model(df, forecast_days=10):
  latest = df.iloc[-1]
  close = latest["Close"]
  atr = latest["ATR"] if pd.notnull(latest["ATR"]) else close * 0.03
  rsi = latest["RSI"] if pd.notnull(latest["RSI"]) else 50
  macd_hist = latest["MACD_Hist"] if pd.notnull(latest["MACD_Hist"]) else 0
  sma20 = latest["SMA_20"] if pd.notnull(latest["SMA_20"]) else close
  vwap = latest["VWAP"] if pd.notnull(latest["VWAP"]) else close

  # Linear slope of last 14 sessions
  recent_closes = df["Close"].tail(14).values
  x = np.arange(len(recent_closes))
  slope, _ = np.polyfit(x, recent_closes, 1)

  # Momentum weighting based on indicators
  momentum_modifier = 1.0
  if macd_hist > 0:
    momentum_modifier += 0.25
  else:
    momentum_modifier -= 0.25

  if rsi > 70:
    momentum_modifier -= 0.4  # Overbought pullback bias
  elif rsi < 30:
    momentum_modifier += 0.4  # Oversold mean-reversion bounce

  if close > vwap:
    momentum_modifier += 0.15
  else:
    momentum_modifier -= 0.15

  daily_vector = (slope * 0.5) + (macd_hist * 0.15 * momentum_modifier)

  # Build forward curve
  last_date = df.index[-1]
  future_dates = [
      last_date + timedelta(days=i + 1) for i in range(forecast_days)
  ]

  projected_prices = []
  upper_confidence = []
  lower_confidence = []

  curr_price = close
  for i in range(1, forecast_days + 1):
    curr_price += daily_vector
    projected_prices.append(curr_price)
    confidence_spread = atr * np.sqrt(i) * 0.5
    upper_confidence.append(curr_price + confidence_spread)
    lower_confidence.append(curr_price - confidence_spread)

  forecast_df = pd.DataFrame(
      {
          "Predicted Path": projected_prices,
          "Upper Target (ATR)": upper_confidence,
          "Lower Support (ATR)": lower_confidence,
      },
      index=future_dates,
  )

  # Determine target trade parameters
  if rsi < 40:
    buy_target = min(
        close, latest["BB_Lower"] if pd.notnull(latest["BB_Lower"]) else close
    )
  else:
    buy_target = max(sma20, close - (0.75 * atr))

  sell_target = max(projected_prices[-1], close + (2.0 * atr))
  stop_loss = buy_target - (1.5 * atr)

  return forecast_df, buy_target, sell_target, stop_loss, daily_vector


# ---------------------------------------------------------
# Dashboard Rendering Engine
# ---------------------------------------------------------
def render_full_dashboard(df, ticker_name):
  latest = df.iloc[-1]
  prev = df.iloc[-2]

  # Row 1: Core Metrics
  c1, c2, c3, c4, c5, c6 = st.columns(6)
  c1.metric(
      "Current Price",
      f"${latest['Close']:,.2f}",
      f"{(latest['Close']-prev['Close']):,.2f}",
  )
  c2.metric(
      "VWAP",
      f"${latest['VWAP']:,.2f}" if pd.notnull(latest["VWAP"]) else "N/A",
  )
  c3.metric(
      "14-Period RSI",
      f"{latest['RSI']:.1f}" if pd.notnull(latest["RSI"]) else "N/A",
  )
  c4.metric(
      "Bollinger %B",
      f"{latest['BB_Percent']:.2f}"
      if pd.notnull(latest["BB_Percent"])
      else "N/A",
  )
  c5.metric(
      "RVOL", f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A"
  )
  c6.metric(
      "14-Period ATR",
      f"${latest['ATR']:,.2f}" if pd.notnull(latest["ATR"]) else "N/A",
  )

  # Institutional Signal Alerts
  st.subheader("🎯 Active Market Diagnostics")
  insights = []
  if latest["Close"] > latest["VWAP"]:
    insights.append(
        "🟢 **Bullish VWAP:** Price trading above institutional baseline."
    )
  else:
    insights.append(
        "🔴 **Bearish VWAP:** Price trading below institutional baseline."
    )

  if latest["RSI"] >= 70:
    insights.append(
        "⚠️ **Overbought RSI:** Increased probability of mean-reversion"
        " pullback."
    )
  elif latest["RSI"] <= 30:
    insights.append(
        "🟢 **Oversold RSI:** Higher probability of reversal bounce."
    )

  if latest["BB_Width"] <= df["BB_Width"].rolling(20).min().iloc[-1]:
    insights.append(
        "⚡ **Vol Squeeze:** Narrow Bollinger Bands indicate impending"
        " expansion."
    )

  st.info(" | ".join(insights))

  # Price & Momentum Charts
  st.subheader(f"Price Action vs. VWAP & Volatility Bands ({ticker_name})")
  st.line_chart(df[["Close", "VWAP", "BB_Upper", "BB_Lower"]])

  st.subheader("MACD Momentum Acceleration")
  st.bar_chart(df["MACD_Hist"])

  # ---------------------------------------------------------
  # Automated Predictive Analysis Section
  # ---------------------------------------------------------
  forecast_df, buy_target, sell_target, stop_loss, daily_vector = (
      generate_predictive_model(df)
  )

  st.markdown("---")
  st.header("🔮 10-Period Predictive Trajectory & Target Levels")

  with st.expander(
      "ℹ️ Methodology: How Factors Drive This Predictive Analysis",
      expanded=True,
  ):
    st.markdown("""
        The forward forecast model synthesizes five primary technical vectors to project price trajectory and confidence bounds:
        1. **14-Session Linear Slope Vector:** Establishes the core directional baseline momentum.
        2. **MACD Histogram Velocity:** Accelerates or decelerates the projected daily slope depending on momentum expansion.
        3. **RSI Mean-Reversion Weight:** Applies counter-trend decay when RSI reaches extreme overbought (>70) or oversold (<30) territories.
        4. **VWAP Position Bias:** Applies positive or negative drift based on whether buyers or sellers control the institutional VWAP benchmark.
        5. **ATR Volatility Bands:** Expands upper and lower confidence limits ($\pm 1.5 \times \text{ATR}$) to outline statistical risk boundaries.
        """)

  # Target Price Recommendation Cards
  t1, t2, t3, t4 = st.columns(4)
  t1.metric(
      "Optimal Buy Target",
      f"${buy_target:,.2f}",
      help="Target entry near key moving average/VWAP support or oversold lower band.",
  )
  t2.metric(
      "Take Profit Target",
      f"${sell_target:,.2f}",
      help="Projected resistance target based on 2x ATR expansion.",
  )
  t3.metric(
      "Recommended Stop Loss",
      f"${stop_loss:,.2f}",
      help="Risk invalidation limit set at 1.5x ATR below entry.",
  )
  t4.metric(
      "Projected Daily Vector",
      f"${daily_vector:+,.2f}/period",
      help="Estimated daily momentum slope.",
  )

  st.subheader("Forecasted Path & Risk Bands")
  st.line_chart(forecast_df)


# ---------------------------------------------------------
# App Layout: Stocks vs Crypto Tabs
# ---------------------------------------------------------
tab_stocks, tab_crypto = st.tabs(["📊 Stock Scanner", "🪙 Crypto Scanner"])

# TAB 1: STOCKS (Default: RUM)
with tab_stocks:
  stock_ticker = st.text_input(
      "Enter Stock Ticker", value="RUM", key="stock_input"
  ).upper()
  if stock_ticker:
    stock_data = yf.Ticker(stock_ticker).history(period=timeframe)
    if not stock_data.empty:
      df_processed = compute_all_indicators(stock_data)
      render_full_dashboard(df_processed, stock_ticker)
    else:
      st.error(f"Could not retrieve stock data for symbol: {stock_ticker}")

# TAB 2: CRYPTO (Default: O40092-USD)
with tab_crypto:
  st.caption(
      "Crypto markets trade 24/7. Symbol pairs require `-USD` extension."
  )
  crypto_preset = st.selectbox(
      "Select Crypto Asset",
      ["O40092-USD", "BTC-USD", "ETH-USD", "SOL-USD", "Custom Input"],
      index=0,
  )

  if crypto_preset == "Custom Input":
    crypto_ticker = st.text_input(
        "Custom Pair (e.g. ADA-USD)", value="ADA-USD", key="crypto_input"
    ).upper()
  else:
    crypto_ticker = crypto_preset

  if crypto_ticker:
    crypto_data = yf.Ticker(crypto_ticker).history(period=timeframe)
    if not crypto_data.empty:
      df_crypto_processed = compute_all_indicators(crypto_data)
      render_full_dashboard(df_crypto_processed, crypto_ticker)
    else:
      st.error(
          f"Could not retrieve crypto data for pair: {crypto_ticker}. Ensure it"
          " ends with `-USD`."
      )
