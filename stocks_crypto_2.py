from datetime import timedelta
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Institutional Scanner & Predictive Engine", layout="wide"
)
st.title("📈 Institutional Trading & Predictive Analytics")

# ---------------------------------------------------------
# Sidebar Controls & Expanded Cheat Sheet
# ---------------------------------------------------------
st.sidebar.header("Global Controls")
timeframe_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
timeframe = st.sidebar.selectbox("Analysis Horizon", timeframe_options, index=4)

st.sidebar.markdown("---")
with st.sidebar.expander(
    "📖 Indicator Cheat Sheet (Beginner Friendly)", expanded=False
):
    st.markdown("""
    * **RSI (Relative Strength Index):** Measures speed of price changes (0–100).
        * `>70`: **Overbought** (Price ran up too fast, potential pullback ahead).
        * `<30`: **Oversold** (Price dropped too hard, potential bargain bounce).
    * **VWAP (Volume-Weighted Average Price):** The average price paid by big institutions throughout the day.
        * Price **above VWAP** = Buyers are in control (Bullish).
        * Price **below VWAP** = Sellers are in control (Bearish).
    * **SMA 20 & 50 (Simple Moving Averages):** Smooth lines showing 20-day or 50-day average price trends.
    * **Bollinger Bands (%B):** Volatility envelopes around price. Upper = Resistance, Lower = Support.
    * **MACD Hist (Histogram):** Green bars = Buying momentum; Red bars = Selling momentum.
    * **ATR (Average True Range):** The expected daily dollar swing size (helps set realistic stop losses).
    * **RVOL (Relative Volume):** Compares today's volume to normal volume (`>1.5x` means institutional big money is moving the stock).
    """)


# ---------------------------------------------------------
# Indicators Engine
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

    # Bollinger Bands
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


# ---------------------------------------------------------
# Advanced Predictive Model Engine
# ---------------------------------------------------------
def generate_predictive_model(df, forecast_days=15):
    latest = df.iloc[-1]
    close = latest["Close"]
    atr = latest["ATR"] if pd.notnull(latest["ATR"]) else close * 0.03
    rsi = latest["RSI"] if pd.notnull(latest["RSI"]) else 50
    macd_hist = latest["MACD_Hist"] if pd.notnull(latest["MACD_Hist"]) else 0
    vwap = latest["VWAP"] if pd.notnull(latest["VWAP"]) else close

    recent_closes = df["Close"].tail(14).values
    x = np.arange(len(recent_closes))
    slope, _ = np.polyfit(x, recent_closes, 1)

    momentum_modifier = 1.0
    if macd_hist > 0:
        momentum_modifier += 0.25
    else:
        momentum_modifier -= 0.25

    if rsi > 70:
        momentum_modifier -= 0.4
    elif rsi < 30:
        momentum_modifier += 0.4

    if close > vwap:
        momentum_modifier += 0.15
    else:
        momentum_modifier -= 0.15

    daily_vector = (slope * 0.5) + (macd_hist * 0.15 * momentum_modifier)

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

    buy_target = min(
        close, latest["BB_Lower"] if pd.notnull(latest["BB_Lower"]) else close
    )
    sell_target = max(projected_prices[-1], close + (2.0 * atr))
    stop_loss = buy_target - (1.5 * atr)

    return forecast_df, buy_target, sell_target, stop_loss, daily_vector


# ---------------------------------------------------------
# News Feed Helper
# ---------------------------------------------------------
def analyze_headline_sentiment(title):
    title_lower = title.lower()
    bullish_keywords = [
        "beat",
        "surged",
        "surge",
        "record",
        "growth",
        "soar",
        "soared",
        "jump",
        "upgraded",
        "upgrade",
        "gain",
        "gains",
        "bull",
        "bullish",
        "profit",
        "rally",
        "highs",
        "outperform",
        "buy",
        "expansion",
        "partnership",
        "success",
    ]
    bearish_keywords = [
        "miss",
        "missed",
        "drop",
        "dropped",
        "fall",
        "fell",
        "plunge",
        "plunged",
        "downgraded",
        "downgrade",
        "loss",
        "losses",
        "bear",
        "bearish",
        "lawsuit",
        "investigation",
        "decline",
        "warning",
        "slump",
        "cuts",
        "slash",
        "probe",
    ]

    bull_count = sum(1 for word in bullish_keywords if word in title_lower)
    bear_count = sum(1 for word in bearish_keywords if word in title_lower)

    if bull_count > bear_count:
        return "🟢 **(GOOD NEWS)**"
    elif bear_count > bull_count:
        return "🔴 **(BAD NEWS)**"
    else:
        return "⚪ **(NEUTRAL / INFORMATIONAL)**"


def render_news_feed(ticker_obj, ticker_name):
    st.markdown("---")
    st.header(f"📰 Recent Market News & Headline Sentiment ({ticker_name})")
    try:
        news_items = ticker_obj.news
        valid_articles = 0
        if news_items:
            for item in news_items:
                content = (
                    item.get("content", {})
                    if isinstance(item.get("content"), dict)
                    else item
                )
                title = content.get("title") or item.get("title")
                link = item.get("link") or content.get("link")
                if (
                    not link
                    and "clickThroughUrl" in content
                    and content["clickThroughUrl"]
                ):
                    link = content["clickThroughUrl"].get("url")
                if (
                    not link
                    and "canonicalUrl" in content
                    and content["canonicalUrl"]
                ):
                    link = content["canonicalUrl"].get("url")

                provider = content.get("provider") or item.get("publisher")
                publisher = (
                    provider.get("displayName")
                    if isinstance(provider, dict)
                    else (provider or "Market Source")
                )

                if title and link:
                    sentiment_tag = analyze_headline_sentiment(title)
                    st.markdown(
                        f"{sentiment_tag} **[{title}]({link})** — *{publisher}*"
                    )
                    valid_articles += 1
                if valid_articles >= 6:
                    break

        if valid_articles == 0:
            st.info(
                f"No recent active news found specifically for **{ticker_name}**."
            )
    except Exception as e:
        st.info(f"Could not load news feed: {e}")


# ---------------------------------------------------------
# Main UI Dashboard Renderer
# ---------------------------------------------------------
def render_full_dashboard(df, ticker_name, asset_type, ticker_obj):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    fmt = "{:,.3f}" if asset_type == "Crypto" else "{:,.2f}"

    # Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "Current Price",
        f"${fmt.format(latest['Close'])}",
        f"{fmt.format(latest['Close']-prev['Close'])}",
    )
    c2.metric(
        "VWAP (Inst. Benchmark)",
        f"${fmt.format(latest['VWAP'])}"
        if pd.notnull(latest["VWAP"])
        else "N/A",
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
        "RVOL (Volume Multiplier)",
        f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A",
    )
    c6.metric(
        "14-Period ATR (Daily Range)",
        f"${fmt.format(latest['ATR'])}" if pd.notnull(latest["ATR"]) else "N/A",
    )

    # Action Recommendation
    st.subheader("🚦 Actionable Trade Recommendation")
    bull_points = 0
    bear_points = 0

    if latest["Close"] > latest["VWAP"]:
        bull_points += 1
    else:
        bear_points += 1
    if latest["RSI"] < 40:
        bull_points += 2
    elif latest["RSI"] > 70:
        bear_points += 2
    if latest["MACD_Hist"] > 0:
        bull_points += 1
    else:
        bear_points += 1
    if latest["Close"] > latest["SMA_20"]:
        bull_points += 1
    else:
        bear_points += 1

    if bull_points >= 4:
        st.success(
            "🟢 **EXECUTIVE ACTION: BUY / ACCUMULATE NOW**\n\n"
            "**Why:** Strong confluence of bullish signals. Price is supported by VWAP/20-SMA with positive MACD momentum."
        )
    elif bear_points >= 4:
        st.error(
            "🔴 **EXECUTIVE ACTION: SELL / TAKE PROFITS NOW**\n\n"
            "**Why:** Heavy overhead resistance detected. Asset is trading below VWAP baseline with decelerating momentum."
        )
    else:
        st.warning(
            "🟡 **EXECUTIVE ACTION: WAIT / HOLD (NO CLEAR EDGE RIGHT NOW)**\n\n"
            "**Why:** Indicators are showing a tug-of-war between buyers and sellers. Momentum is neutral or consolidating."
        )

    # 1. Price vs VWAP Chart
    st.subheader(f"Price Action vs. VWAP & Volatility Bands ({ticker_name})")
    chart_data = df[["Close", "VWAP", "BB_Upper", "BB_Lower"]].dropna()
    st.line_chart(chart_data)

    # 2. RSI Chart
    st.subheader("RSI Momentum (14-Period)")
    st.line_chart(df[["RSI"]].dropna())

    # 3. MACD Colored Histogram using Altair
    st.subheader(
        "MACD Momentum Acceleration (Green = Bullish Momentum | Red = Bearish Momentum)"
    )
    macd_df = df[["MACD_Hist"]].reset_index()
    macd_df["Color"] = np.where(
        macd_df["MACD_Hist"] >= 0, "Bullish (Green)", "Bearish (Red)"
    )

    macd_chart = (
        alt.Chart(macd_df)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("MACD_Hist:Q", title="MACD Histogram"),
            color=alt.Color(
                "Color:N",
                scale=alt.Scale(
                    domain=["Bullish (Green)", "Bearish (Red)"],
                    range=["#22C55E", "#EF4444"],
                ),
            ),
        )
        .properties(height=250)
    )
    st.altair_chart(macd_chart, use_container_width=True)

    # 4. Predictive Trajectory Model
    st.markdown("---")
    st.header("🔮 Interactive Predictive Trajectory & Target Levels")

    forecast_days = st.slider(
        "Adjust Forecast Horizon (Days):",
        min_value=5,
        max_value=60,
        value=15,
        key=f"slider_{ticker_name}",
    )
    forecast_df_custom, buy_t, sell_t, stop_l, d_vec = generate_predictive_model(
        df, forecast_days=forecast_days
    )

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimal Buy Target", f"${fmt.format(buy_t)}")
    t2.metric("Take Profit Target", f"${fmt.format(sell_t)}")
    t3.metric("Recommended Stop Loss", f"${fmt.format(stop_l)}")
    t4.metric("Projected Change Per Period", f"${fmt.format(d_vec)}/period")

    pred_chart_data = forecast_df_custom[
        ["Predicted Path", "Upper Target (ATR)", "Lower Support (ATR)"]
    ]
    st.line_chart(pred_chart_data)

    # Quantitative Breakdown Table
    st.markdown("---")
    st.header(f"📊 Quantitative Analyst Breakdown ({ticker_name})")
    rsi_val = latest["RSI"] if pd.notnull(latest["RSI"]) else 50
    rsi_simple = (
        "Gaining Strength"
        if rsi_val > 55
        else ("Bargain Zone" if rsi_val < 35 else "Neutral")
    )
    rvol_val = latest["RVOL"] if pd.notnull(latest["RVOL"]) else 1.0

    quant_matrix = pd.DataFrame({
        "Market Metric": [
            "Price Momentum (RSI)",
            "Trading Volume (RVOL)",
            "Institutional VWAP Line",
        ],
        "Current Reading": [
            f"{rsi_val:.1f} Score",
            f"{rvol_val:.2f}x Normal",
            f"${fmt.format(latest['VWAP'])} Baseline",
        ],
        "What It Means": [
            rsi_simple,
            "High Volume Active" if rvol_val > 1.25 else "Normal Volume",
            "Bullish (Above Line)"
            if latest["Close"] > latest["VWAP"]
            else "Bearish (Below Line)",
        ],
    })
    st.table(quant_matrix)

    render_news_feed(ticker_obj, ticker_name)


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
tab_stocks, tab_crypto = st.tabs(["📊 Stock Scanner", "🪙 Crypto Scanner"])

with tab_stocks:
    stock_ticker = st.text_input(
        "Enter Stock Ticker", value="NVDA", key="stock_input"
    ).upper()
    if stock_ticker:
        st_obj = yf.Ticker(stock_ticker)
        stock_data = st_obj.history(period=timeframe)
        if not stock_data.empty:
            df_processed = compute_all_indicators(stock_data)
            render_full_dashboard(
                df_processed,
                stock_ticker,
                asset_type="Stock",
                ticker_obj=st_obj,
            )
        else:
            st.error(
                f"Could not retrieve stock data for symbol: {stock_ticker}"
            )

with tab_crypto:
    st.caption("Crypto markets trade 24/7. Currency pair ends with `-USD`.")
    crypto_preset = st.selectbox(
        "Select Crypto Asset",
        ["BTC-USD", "ETH-USD", "SOL-USD", "Custom Input"],
        index=0,
    )
    crypto_ticker = (
        st.text_input(
            "Custom Pair (e.g. ADA-USD)", value="ADA-USD", key="crypto_input"
        ).upper()
        if crypto_preset == "Custom Input"
        else crypto_preset
    )

    if crypto_ticker:
        cr_obj = yf.Ticker(crypto_ticker)
        crypto_data = cr_obj.history(period=timeframe)
        if not crypto_data.empty:
            df_crypto_processed = compute_all_indicators(crypto_data)
            render_full_dashboard(
                df_crypto_processed,
                crypto_ticker,
                asset_type="Crypto",
                ticker_obj=cr_obj,
            )
        else:
            st.error(
                f"Could not retrieve crypto data for pair: {crypto_ticker}."
            )
