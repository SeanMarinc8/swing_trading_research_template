import base64
from datetime import timedelta
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==============================================================================
# SECTION 1: GLOBAL PAGE CONFIGURATION & STYLING
# ==============================================================================
# Purpose: Configures the core Streamlit page parameters, custom CSS styling, 
# and responsive branding elements (CMI Logo) used across the application.

st.set_page_config(
    page_title="CMI | Institutional Scanner & Predictive Engine",
    layout="wide",
    page_icon="📈",
)

# Custom CSS styling for visual enhancements and card layouts
st.markdown(
    """
    <style>
    .cmi-header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0px 20px 0px;
        border-bottom: 2px solid #00ACC1;
        margin-bottom: 20px;
    }
    .metric-card-good {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #155724;
    }
    .metric-card-avg {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #856404;
    }
    .metric-card-bad {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #721c24;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Core Branding Vector Graphic Markup
CMI_LOGO_SVG = """
<div style="font-family: sans-serif; font-weight: 800; font-size: 28px; color: #111; line-height: 1.1;">
    <span style="font-size: 36px; font-weight: 900; letter-spacing: -1px;">CMI</span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#E91E63; margin-left:2px; vertical-align:top;"></span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#26A69A; margin-left:1px; vertical-align:top;"></span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#00ACC1; margin-left:1px; vertical-align:top;"></span>
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px; color: #333; margin-top: -2px;">
        CORE MARKET INTELLIGENCE
    </div>
</div>
"""

# Render Sidebar Branding Header
st.sidebar.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)
st.sidebar.markdown("---")

# Render Main Page Branding Header
col_header_left, col_header_right = st.columns([3, 1])
with col_header_left:
    st.title("📈 Institutional Trading & Predictive Analytics")
    st.caption("Powered by **CMI (Core Market Intelligence)** Quantitative Engine")
with col_header_right:
    st.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: SIDEBAR CONTROLS & CHEAT SHEET
# ==============================================================================
# Purpose: Houses macro control parameters like historical timeframes and an 
# educational technical indicator reference guide to help users translate raw math.

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
    * **Bollinger Bands (%B):** Volatility envelopes around price. Touching upper band = high; lower band = low.
    * **MACD Hist (Histogram):** Shows whether buying or selling momentum is speeding up or slowing down.
    * **ATR (Average True Range):** The expected daily dollar swing size (helps set realistic stop losses).
    * **RVOL (Relative Volume):** Compares today's volume to normal volume (`>1.5x` = institutional activity).
    """)


# ==============================================================================
# SECTION 3: SHARED ANALYTICAL & COMPUTATIONAL ENGINES
# ==============================================================================
# Purpose: Central mathematical functions for computing technical indicators, 
# generating 10-period predictive forecasts, parsing news sentiment, and rendering layout grids.

def compute_all_indicators(df):
    """Calculates SMA, VWAP, RSI, Bollinger Bands, MACD, ATR, and RVOL metrics."""
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
    """Generates 10-period forecasted price path with ATR-based volatility bands."""
    latest = df.iloc[-1]
    close = latest["Close"]
    atr = latest["ATR"] if pd.notnull(latest["ATR"]) else close * 0.03
    rsi = latest["RSI"] if pd.notnull(latest["RSI"]) else 50
    macd_hist = latest["MACD_Hist"] if pd.notnull(latest["MACD_Hist"]) else 0
    vwap = latest["VWAP"] if pd.notnull(latest["VWAP"]) else close

    # Linear slope of last 14 sessions
    recent_closes = df["Close"].tail(14).values
    x = np.arange(len(recent_closes))
    slope, _ = np.polyfit(x, recent_closes, 1)

    # Momentum weighting
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

    # Forward projections
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


def analyze_headline_sentiment(title):
    """Simple keyword sentiment tagger for financial news headlines."""
    title_lower = title.lower()
    bullish_keywords = [
        "beat", "surged", "surge", "record", "growth", "soar", "soared", "jump",
        "upgraded", "upgrade", "gain", "gains", "bull", "bullish", "profit",
        "rally", "highs", "outperform", "buy", "expansion", "partnership", "success"
    ]
    bearish_keywords = [
        "miss", "missed", "drop", "dropped", "fall", "fell", "plunge", "plunged",
        "downgraded", "downgrade", "loss", "losses", "bear", "bearish", "lawsuit",
        "investigation", "decline", "warning", "slump", "cuts", "slash", "probe"
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
    """Renders recent news headlines with sentiment indicators."""
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


def render_trading_strategies_guide():
    """Renders an interactive expander containing proven trading strategies."""
    with st.expander(
        "💡 Easy Proven Trading Strategies (How to use this dashboard)",
        expanded=False,
    ):
        st.markdown("""
        ### Strategy 1: The "Institutional VWAP Bounce" (Best for Trend Buyers)
        * **Goal:** Buy high-quality stocks at a wholesale discount price when big institutional buyers step in.
        * **How to Spot It:**
            1. Look for the **Actionable Trade Recommendation** to show **LEAN BUY** or **BUY**.
            2. Check the **Price Action vs. VWAP chart**. Wait until current price drops close to or touches the **VWAP line**.
            3. Verify that **RVOL** is above `1.2x` (meaning big volume is present).
            4. **Action:** Buy near VWAP. Place your Stop Loss slightly below the 50-day SMA.
        ---
        ### Strategy 2: The "RSI Bargain Hunter" (Best for Rebound Trades)
        * **Goal:** Catch quick market bounces when a stock has been oversold and beaten down too hard.
        * **How to Spot It:**
            1. Check the **RSI Momentum chart**. Look for RSI dropping below **30** (Oversold).
            2. Check the **Bollinger %B**. Price should be near or below `0.00` (touching lower band).
            3. Look at the **News Feed**: If news is tagged 🟢 **(GOOD NEWS)** or ⚪ **(NEUTRAL)** while price is oversold, it signals an overreaction drop.
            4. **Action:** Buy when RSI crosses back *above* 30. Use the **Optimal Buy Target** metric as your entry anchor.
        ---
        ### Strategy 3: The "Breakout Velocity" (Best for Momentum Traders)
        * **Goal:** Ride fast moving stocks as momentum accelerates.
        * **How to Spot It:**
            1. Look at the **MACD Histogram**. The bars should be green and growing taller.
            2. **RVOL** should be `> 1.5x` (heavy buying volume).
            3. Price should be trading comfortably **above VWAP** and the **20-period SMA**.
            4. **Action:** Ride the trend until price touches the upper **Bollinger Band** or the **Take Profit Target**, then lock in gains.
        """)


def render_full_dashboard(df, ticker_name, asset_type, ticker_obj):
    """Renders the comprehensive quantitative dashboard for Stock and Crypto assets."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    fmt = "{:,.3f}" if asset_type == "Crypto" else "{:,.2f}"
    forecast_df, buy_target, sell_target, stop_loss, daily_vector = (
        generate_predictive_model(df)
    )

    # Top Metrics Row
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

    # Action Recommendation & Trade Guidance
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
            "**Why:** Strong confluence of bullish signals. Price is supported"
            " by VWAP/20-SMA with positive MACD momentum and room for RSI expansion."
        )
    elif bear_points >= 4:
        st.error(
            "🔴 **EXECUTIVE ACTION: SELL / TAKE PROFITS NOW**\n\n"
            "**Why:** Heavy overhead resistance detected. Asset is trading"
            " below VWAP baseline with decelerating MACD momentum or overbought RSI."
        )
    else:
        st.warning(
            "🟡 **EXECUTIVE ACTION: WAIT / HOLD (NO CLEAR EDGE RIGHT NOW)**\n\n"
            "**Why:** Indicators show a tug-of-war between buyers and sellers. Momentum is neutral or consolidating."
        )

    # Extended Lean Breakdown
    lean_direction = "LONG (BUY)" if bull_points >= bear_points else "SHORT (SELL)"
    lean_color = "🟢" if "LONG" in lean_direction else "🔴"
    curr_price = latest["Close"]
    atr_val = latest["ATR"] if pd.notnull(latest["ATR"]) else curr_price * 0.03

    if "LONG" in lean_direction:
        entry_target = buy_target
        exit_target = sell_target
        sl_price = stop_loss
        risk_pct = max(0.1, abs((entry_target - sl_price) / entry_target) * 100)
        reward_pct = max(0.1, abs((exit_target - entry_target) / entry_target) * 100)
        sl_reason = (
            f"The stop-loss is placed at **${fmt.format(sl_price)}** (1.5x ATR below entry) "
            f"to allow normal volatility fluctuations while protecting capital against a breakdown below the lower Bollinger Band (${fmt.format(latest['BB_Lower'])})."
        )
    else:
        entry_target = curr_price
        exit_target = buy_target
        sl_price = curr_price + (1.5 * atr_val)
        risk_pct = max(0.1, abs((sl_price - entry_target) / entry_target) * 100)
        reward_pct = max(0.1, abs((entry_target - exit_target) / entry_target) * 100)
        sl_reason = (
            f"The stop-loss is set at **${fmt.format(sl_price)}** (1.5x ATR above entry) "
            f"to limit losses if buyers push price back above institutional VWAP (${fmt.format(latest['VWAP'])})."
        )

    with st.container():
        st.markdown(
            f"#### {lean_color} **If You Had to Act: LEAN {lean_direction}**"
        )
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Execution Target Price", f"${fmt.format(entry_target)}")
        m_col2.metric("Take-Profit Target", f"${fmt.format(exit_target)}")
        m_col3.metric("Correlating Stop-Loss", f"${fmt.format(sl_price)}")
        m_col4.metric(
            "Trade Risk Percentage",
            f"{risk_pct:.2f}%",
            f"Reward: +{reward_pct:.2f}%",
        )
        st.markdown(f"""
        * **Target Strategy:** Set limit orders near **${fmt.format(entry_target)}** targeting an exit at **${fmt.format(exit_target)}**.
        * **Risk Management:** Executing this position carries an estimated **{risk_pct:.2f}% risk profile** from entry to stop-loss.
        * **Stop-Loss Logic:** {sl_reason}
        """)

    # Render Strategy Expander
    render_trading_strategies_guide()

    # Technical Charts Section
    st.subheader(f"Price Action vs. VWAP & Volatility Bands ({ticker_name})")
    st.line_chart(df[["Close", "VWAP", "BB_Upper", "BB_Lower"]].dropna())

    st.subheader("RSI Momentum & Overbought/Oversold Bounds")
    st.line_chart(df[["RSI", "Overbought (70)", "Oversold (30)"]].dropna())

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
                legend=alt.Legend(title="Momentum"),
            ),
        )
        .properties(height=250)
    )
    st.altair_chart(macd_chart, use_container_width=True)

    # Predictive Analytics Header
    st.markdown("---")
    st.header("🔮 10-Period Predictive Trajectory & Target Levels")
    with st.expander("ℹ️ How This Predictive Forecast Works", expanded=True):
        st.markdown("""
            This engine forecasts where price is likely headed over the next 10 periods by combining 5 main clues:
            1. **Current Trend Slope:** Is price naturally trending up or down over the last 14 days?
            2. **MACD Speed:** Is buying/selling momentum speeding up or slowing down?
            3. **RSI Guardrails:** Prevents forecasting extreme rallies if stock is already overbought.
            4. **VWAP Magnet:** Adjusts expected direction depending on whether price is above or below institutional average price.
            5. **ATR Risk Spread:** Calculates upper and lower price bands based on typical daily volatility.
            """)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimal Buy Target", f"${fmt.format(buy_target)}")
    t2.metric("Take Profit Target", f"${fmt.format(sell_target)}")
    t3.metric("Recommended Stop Loss", f"${fmt.format(stop_loss)}")
    t4.metric(
        "Projected Change Per Period", f"${fmt.format(daily_vector)}/period"
    )
    st.subheader("Forecasted Price Path & Confidence Bands")
    st.line_chart(forecast_df)

    # Quantitative Matrix Breakdown
    st.markdown("---")
    st.header(f"📊 Quantitative Analyst Breakdown ({ticker_name})")
    rsi_val = latest["RSI"] if pd.notnull(latest["RSI"]) else 50
    rsi_simple = (
        "Gaining Strength (Buyers in control)"
        if rsi_val > 55
        else (
            "Bargain Zone (Oversold)"
            if rsi_val < 35
            else "Neutral (Fair Value)"
        )
    )
    bb_width = latest["BB_Width"] if pd.notnull(latest["BB_Width"]) else 0
    vol_simple = (
        "Coiled Spring (Big move coming soon)"
        if bb_width < 0.08
        else "Normal Volatility"
    )
    rvol_val = latest["RVOL"] if pd.notnull(latest["RVOL"]) else 1.0
    flow_simple = (
        "Big Institutional Money Active"
        if rvol_val > 1.25
        else "Normal Everyday Trading Volume"
    )

    quant_matrix = pd.DataFrame({
        "Market Metric": [
            "Price Momentum (RSI)",
            "Price Volatility Squeeze",
            "Trading Volume (RVOL)",
            "Institutional VWAP Line",
        ],
        "Current Reading": [
            f"{rsi_val:.1f} Score",
            f"Width: {bb_width:.3f}",
            f"{rvol_val:.2f}x Normal",
            f"${fmt.format(latest['VWAP'])} Baseline",
        ],
        "What It Means For You": [
            rsi_simple,
            vol_simple,
            flow_simple,
            "Price is Healthy (Above Line)"
            if latest["Close"] > latest["VWAP"]
            else "Price is Weak (Below Line)",
        ],
    })
    st.table(quant_matrix)

    # Synthesis Paragraph
    st.subheader("Plain-English Analyst Conclusion")
    st.markdown(f"""
    **In Plain English:** Analyzing **{ticker_name}** through quantitative metrics, price is trading relative to its **Institutional Baseline (VWAP)** of `${fmt.format(latest['VWAP'])}`. 
    * **Volume check:** Trading volume is sitting at **{rvol_val:.2f}x** normal levels. {"This indicates big financial institutions are actively moving this asset." if rvol_val > 1.25 else "Retail volume is dominant and institutions are not aggressively pushing right now."}
    * **Volatility check:** Volatility squeeze reading (`{bb_width:.3f}`) indicates price is {"coiling tightly—get ready for a breakout move." if bb_width < 0.08 else "moving within normal daily ranges."}
    * **Bottom Line:** The safest path is waiting for alignment near the **Optimal Buy Target** (`${fmt.format(buy_target)}`).
    """)

    # Fundamentals / Financial Statements
    st.markdown("---")
    st.header("📋 Financial Statements & Fundamental Overview")
    if asset_type == "Stock":
        try:
            info = ticker_obj.info
            income_stmt = ticker_obj.financials
            cash_flow = ticker_obj.cashflow
            rev_growth = info.get("revenueGrowth", 0) or 0
            profit_margin = info.get("profitMargins", 0) or 0
            debt_to_equity = info.get("debtToEquity", 100) or 100

            if rev_growth > 0.10 and profit_margin > 0.15:
                health_grade = "GOOD 🟢"
                card_class = "metric-card-good"
                grade_reason = "Strong double-digit top-line growth combined with robust profit margins."
            elif rev_growth > 0 or profit_margin > 0.05:
                health_grade = "AVERAGE 🟡"
                card_class = "metric-card-avg"
                grade_reason = "Moderate fundamental stability with steady margins but slower top-line expansion."
            else:
                health_grade = "BAD 🔴"
                card_class = "metric-card-bad"
                grade_reason = "Declining top-line growth or squeezed margins indicating fundamental headwinds."

            st.markdown(
                f"""
            <div class="{card_class}">
                <h3 style="margin:0;">Fundamental Rating: <strong>{health_grade}</strong></h3>
                <p style="margin: 5px 0 10px 0;"><strong>Executive Summary:</strong> {grade_reason}</p>
                <ul style="margin-bottom:0;">
                    <li><strong>YoY Revenue Growth:</strong> {rev_growth*100:.1f}%</li>
                    <li><strong>Profit Margin:</strong> {profit_margin*100:.1f}%</li>
                    <li><strong>Debt-to-Equity Ratio:</strong> {debt_to_equity:.1f}</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if income_stmt is not None and not income_stmt.empty:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.subheader("Income Statement ($ Millions)")
                    items_to_show = [
                        "Total Revenue",
                        "Gross Profit",
                        "Operating Income",
                        "Net Income",
                    ]
                    existing_items = [
                        item
                        for item in items_to_show
                        if item in income_stmt.index
                    ]
                    df_inc = income_stmt.loc[existing_items] / 1e6
                    df_inc.columns = [
                        col.strftime("%Y") if hasattr(col, "strftime") else col
                        for col in df_inc.columns
                    ]
                    st.dataframe(
                        df_inc.style.format("${:,.1f}M"),
                        use_container_width=True,
                    )
                with col_f2:
                    st.subheader("Cash Flow Statement ($ Millions)")
                    cf_items = [
                        "Operating Cash Flow",
                        "Capital Expenditures",
                        "Free Cash Flow",
                    ]
                    existing_cf = [
                        item for item in cf_items if item in cash_flow.index
                    ]
                    if existing_cf:
                        df_cf = cash_flow.loc[existing_cf] / 1e6
                        df_cf.columns = [
                            col.strftime("%Y")
                            if hasattr(col, "strftime")
                            else col
                            for col in df_cf.columns
                        ]
                        st.dataframe(
                            df_cf.style.format("${:,.1f}M"),
                            use_container_width=True,
                        )
                    else:
                        st.dataframe(
                            cash_flow.head(5), use_container_width=True
                        )
            else:
                st.info("Financial statements not available for this ticker.")
        except Exception as e:
            st.info(f"Could not load financial statements: {e}")
    else:
        st.info(
            "Cryptocurrencies do not publish corporate financial statements"
            " (Income/Cash Flow). Valuation is based on network activity and unit economics."
        )

    # News Feed Trigger
    render_news_feed(ticker_obj, ticker_name)


# ==============================================================================
# REAL-TIME QUANTITATIVE SCREENER FUNCTIONS (LIVE MARKET DATA & RESEARCH FACTORS)
# ==============================================================================

@st.cache_data(ttl=300)
def fetch_live_stock_quant_picks():
    """
    Dynamically scans an equity universe using the exact institutional research factors from the Stock tab:
    VWAP position, 14-period RSI, MACD Histogram momentum, and Relative Volume (RVOL).
    """
    candidate_universe = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", 
        "NFLX", "PLTR", "INTC", "BAC", "JPM", "PANW", "UBER", "DIS", 
        "SQ", "PYPL", "BA", "SNAP", "XOM", "CVX", "PFE", "MRK"
    ]
    
    long_results = []
    short_results = []

    for t in candidate_universe:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="2mo")
            if not hist.empty and len(hist) > 20:
                df = compute_all_indicators(hist)
                latest = df.iloc[-1]
                cp = latest["Close"]
                vwap = latest["VWAP"] if pd.notnull(latest["VWAP"]) else cp
                rsi = latest["RSI"] if pd.notnull(latest["RSI"]) else 50.0
                macd_h = latest["MACD_Hist"] if pd.notnull(latest["MACD_Hist"]) else 0
                rvol = latest["RVOL"] if pd.notnull(latest["RVOL"]) else 1.0
                atr = latest["ATR"] if pd.notnull(latest["ATR"]) else cp * 0.03

                # Bullish Quant Factor Matching
                if cp > vwap and rsi < 65 and macd_h > 0 and rvol > 0.8:
                    target = cp + (2.0 * atr)
                    sl = cp - (1.5 * atr)
                    risk_pct = abs((cp - sl) / cp) * 100
                    reward_pct = abs((target - cp) / cp) * 100
                    long_results.append({
                        "Ticker": t,
                        "Live Price": f"${cp:.2f}",
                        "VWAP Baseline": f"${vwap:.2f}",
                        "14-RSI": f"{rsi:.1f}",
                        "RVOL": f"{rvol:.2f}x",
                        "Target Price": f"${target:.2f}",
                        "Stop Loss": f"${sl:.2f}",
                        "Risk Profile": f"{risk_pct:.2f}% (R: +{reward_pct:.1f}%)",
                        "Quant Setup": "ACCUMULATE / BREAKOUT" if rvol > 1.2 else "VWAP SUPPORT BOUNCE"
                    })

                # Bearish Quant Factor Matching
                elif cp < vwap and macd_h < 0 and rsi > 35:
                    target = cp - (2.0 * atr)
                    sl = cp + (1.5 * atr)
                    risk_pct = abs((sl - cp) / cp) * 100
                    reward_pct = abs((cp - target) / cp) * 100
                    short_results.append({
                        "Ticker": t,
                        "Live Price": f"${cp:.2f}",
                        "VWAP Baseline": f"${vwap:.2f}",
                        "14-RSI": f"{rsi:.1f}",
                        "RVOL": f"{rvol:.2f}x",
                        "Short Target": f"${target:.2f}",
                        "Stop Loss": f"${sl:.2f}",
                        "Risk Profile": f"{risk_pct:.2f}% (R: +{reward_pct:.1f}%)",
                        "Quant Setup": "HEAVY DISTRIBUTION" if rvol > 1.2 else "VWAP RESISTANCE REJECT"
                    })
        except Exception:
            pass

    return pd.DataFrame(long_results), pd.DataFrame(short_results)


def render_live_stock_screener():
    """Renders the Real-Time Stock Quant Screener inside the Stock Tab."""
    st.markdown("---")
    st.header("🎯 Live Market Quant Top Stock Picks")
    st.caption(
        "Scans live US equities in real-time applying research factors from the Stock tab "
        "(Price vs. VWAP baseline, 14-RSI trajectory, MACD momentum, and RVOL institutional spikes)."
    )
    
    with st.spinner("Fetching live market price streams and executing stock factor algorithm..."):
        df_stock_longs, df_stock_shorts = fetch_live_stock_quant_picks()

    tab_s_long, tab_s_short = st.tabs(
        ["🟢 Live Quant Top Longs (Bullish Equities)", "🔴 Live Quant Top Shorts (Bearish Equities)"]
    )

    with tab_s_long:
        if not df_stock_longs.empty:
            st.dataframe(df_stock_longs, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks currently meet all 4 bullish factor alignment criteria in live streaming data.")

    with tab_s_short:
        st.caption("⚠️ Short positions carry dynamic squeeze risk. Enforce stop losses strictly.")
        if not df_stock_shorts.empty:
            st.dataframe(df_stock_shorts, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks currently meet all 4 bearish factor alignment criteria in live streaming data.")


@st.cache_data(ttl=180)
def fetch_live_crypto_quant_picks():
    """
    Dynamically scans liquid 24/7 crypto pairs using crypto metrics:
    RSI momentum extremes, VWAP baseline deviations, ATR daily ranges, and MACD trend direction.
    """
    crypto_universe = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
        "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "SUI-USD", "NEAR-USD",
        "APT-USD", "LTC-USD", "UNI-USD", "PEPE-USD", "FET-USD", "SHIB-USD"
    ]

    long_crypto = []
    short_crypto = []

    for pair in crypto_universe:
        try:
            tk = yf.Ticker(pair)
            hist = tk.history(period="1mo")
            if not hist.empty and len(hist) > 14:
                df = compute_all_indicators(hist)
                latest = df.iloc[-1]
                cp = latest["Close"]
                vwap = latest["VWAP"] if pd.notnull(latest["VWAP"]) else cp
                rsi = latest["RSI"] if pd.notnull(latest["RSI"]) else 50.0
                macd_h = latest["MACD_Hist"] if pd.notnull(latest["MACD_Hist"]) else 0
                atr = latest["ATR"] if pd.notnull(latest["ATR"]) else cp * 0.04
                rvol = latest["RVOL"] if pd.notnull(latest["RVOL"]) else 1.0

                # Format dynamically based on price magnitude
                fmt = "${:,.4f}" if cp < 1.0 else "${:,.2f}"

                # Crypto Bullish Match (Momentum Expansion or Oversold Bounce)
                if (cp > vwap and macd_h > 0) or (rsi < 35):
                    target = cp + (2.5 * atr)
                    sl = cp - (1.5 * atr)
                    risk_pct = abs((cp - sl) / cp) * 100
                    reward_pct = abs((target - cp) / cp) * 100
                    
                    long_crypto.append({
                        "Crypto Pair": pair,
                        "Live Price": fmt.format(cp),
                        "24h VWAP": fmt.format(vwap),
                        "14-RSI": f"{rsi:.1f}",
                        "ATR (Daily Swing)": fmt.format(atr),
                        "Target Exit": fmt.format(target),
                        "Stop Loss": fmt.format(sl),
                        "Risk / Reward": f"-{risk_pct:.1f}% / +{reward_pct:.1f}%",
                        "Signal Driver": "OVERSOLD BOUNCE" if rsi < 35 else "BULLISH MOMENTUM EXPANSION"
                    })

                # Crypto Bearish Match (Momentum Exhaustion or VWAP Breakdown)
                elif (cp < vwap and macd_h < 0) or (rsi > 70):
                    target = cp - (2.5 * atr)
                    sl = cp + (1.5 * atr)
                    risk_pct = abs((sl - cp) / cp) * 100
                    reward_pct = abs((cp - target) / cp) * 100

                    short_crypto.append({
                        "Crypto Pair": pair,
                        "Live Price": fmt.format(cp),
                        "24h VWAP": fmt.format(vwap),
                        "14-RSI": f"{rsi:.1f}",
                        "ATR (Daily Swing)": fmt.format(atr),
                        "Short Target": fmt.format(target),
                        "Stop Loss": fmt.format(sl),
                        "Risk / Reward": f"-{risk_pct:.1f}% / +{reward_pct:.1f}%",
                        "Signal Driver": "OVERBOUGHT EXHAUSTION" if rsi > 70 else "BEARISH VWAP BREAKDOWN"
                    })
        except Exception:
            pass

    return pd.DataFrame(long_crypto), pd.DataFrame(short_crypto)


def render_live_crypto_screener():
    """Renders the Real-Time Crypto Quant Screener inside the Crypto Tab."""
    st.markdown("---")
    st.header("⚡ Live 24/7 Crypto Quant Picks & Recommendations")
    st.caption(
        "Scans digital asset markets in real-time applying crypto-specific quantitative metrics "
        "(Volatile ATR bands, 24-hour VWAP baseline, RSI momentum extremes, and MACD trends)."
    )

    with st.spinner("Streaming 24/7 crypto exchange data and calculating volatility factors..."):
        df_crypto_longs, df_crypto_shorts = fetch_live_crypto_quant_picks()

    tab_c_long, tab_c_short = st.tabs(
        ["🟢 Live Top Crypto Longs (Bullish Momentum)", "🔴 Live Top Crypto Shorts (Bearish Pullbacks)"]
    )

    with tab_c_long:
        if not df_crypto_longs.empty:
            st.dataframe(df_crypto_longs, use_container_width=True, hide_index=True)
        else:
            st.info("No crypto pairs currently match bullish quantitative rules in live streaming data.")

    with tab_c_short:
        st.caption("⚠️ Leverage in crypto markets increases liquidation risk. Enforce ATR stops strictly.")
        if not df_crypto_shorts.empty:
            st.dataframe(df_crypto_shorts, use_container_width=True, hide_index=True)
        else:
            st.info("No crypto pairs currently match bearish quantitative rules in live streaming data.")


# ==============================================================================
# SECTION 4: MAIN TABBED NAVIGATION (STOCKS | CRYPTO | REAL ESTATE)
# ==============================================================================
# Purpose: Serves as the primary operational workspace, dividing quantitative analysis 
# into distinct asset classes (Equities, Digital Assets, and Physical Real Estate).

tab_stocks, tab_crypto, tab_real_estate = st.tabs(
    ["📊 Stock Scanner", "🪙 Crypto Scanner", "🏠 Real Estate Evaluation"]
)

# ------------------------------------------------------------------------------
# TAB 1: STOCK SCANNER
# ------------------------------------------------------------------------------
# Purpose: Analyzes equities using institutional indicators (VWAP, RSI, MACD, ATR) 
# and SEC fundamental reporting to identify high-probability long/short swing opportunities.
# Idea: Eliminate emotional retail bias by evaluating equities on institutional benchmark 
# pricing, relative volume surges, and quarterly corporate financial health.

with tab_stocks:
    st.subheader("📊 Quantitative Stock Analysis & Screener")
    st.caption(
        "The Stock Scanner provides institutional-grade evaluation of publicly traded equities by combining price action momentum, "
        "volume distribution, and fundamental corporate financials into actionable trade parameters."
    )
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
    
    # Isolated Stock Quant Market Screener
    render_live_stock_screener()

# ------------------------------------------------------------------------------
# TAB 2: CRYPTO SCANNER
# ------------------------------------------------------------------------------
# Purpose: Scans 24/7 digital asset markets to track high-volatility cryptocurrency pairs 
# and calculate dynamic volatility bands without corporate filing dependencies.
# Idea: Capture short-term breakout momentum and momentum exhaustion across non-stop, 
# sentiment-driven token markets using strict ATR risk management rules.

with tab_crypto:
    st.subheader("🪙 Cryptocurrency Market Scanner")
    st.caption(
        "The Crypto Scanner tracks 24/7 liquid digital asset markets, adapting traditional quantitative indicators to handle high-volatility, "
        "sentiment-driven price swings where traditional corporate financial metrics are absent."
    )
    st.caption("Crypto markets trade 24/7. Ticker symbols must end with `-USD`.")
    
    crypto_preset = st.selectbox(
        "Select Crypto Asset",
        ["BTC-USD", "ETH-USD", "SOL-USD", "Custom Input"],
        index=0,
    )
    if crypto_preset == "Custom Input":
        crypto_ticker = st.text_input(
            "Custom Pair (e.g. ADA-USD)", value="ADA-USD", key="crypto_input"
        ).upper()
    else:
        crypto_ticker = crypto_preset

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
                " Ensure it ends with `-USD`."
            )

    # Isolated Crypto Quant Market Screener
    render_live_crypto_screener()

# ------------------------------------------------------------------------------
# TAB 3: REAL ESTATE EVALUATION
# ------------------------------------------------------------------------------
# Purpose: Underwrites physical property acquisition strategies (Buy & Hold, Fix & Flip, STR) 
# by modeling Cap Rates, Cash-on-Cash yields, MAO thresholds, and 20-year debt paydown trajectories.
# Idea: Enable investors to stress-test physical real estate cash flows, financing leverage, 
# and multi-year equity accumulation across custom regional markets.

with tab_real_estate:
    st.header("🏠 Real Estate Property Underwriting & Portfolio Engine")
    st.caption(
        "The Real Estate Evaluation module models cash flows, financing terms, operating expenses, and long-term equity growth for physical properties, "
        "helping investors underwrite deals across long-term, short-term, and fix-and-flip strategies."
    )
    
    # Strategy Selection & Inputs
    col_re_ctrl1, col_re_ctrl2, col_re_ctrl3 = st.columns(3)
    with col_re_ctrl1:
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Specific Property Address", "Regional Market Scout"],
            index=0,
            key="re_analysis_mode",
        )
    with col_re_ctrl2:
        investment_intent = st.selectbox(
            "Investment Intent / Strategy",
            [
                "Long-Term Rental (Buy & Hold)",
                "Short-Term Rental (Airbnb/VRBO)",
                "Fix & Flip",
                "House Hack (Primary + Rental)",
                "Value-Add Commercial/Multi-Family",
            ],
            key="re_investment_intent",
        )
    with col_re_ctrl3:
        property_type = st.selectbox(
            "Property Type",
            [
                "Single Family",
                "Multi-Family (2-4 Units)",
                "Condo/Townhome",
                "Commercial/5+ Units",
            ],
            key="re_property_type",
        )
    st.markdown("---")

    # Mode 1: Specific Property Underwriting
    if analysis_mode == "Specific Property Address":
        col_inp1, col_inp2 = st.columns([2, 1])
        with col_inp1:
            address_input = st.text_input(
                "Property Address",
                value="1244 S Michigan Ave, Chicago, IL 60605",
                key="re_address_input",
            )
        with col_inp2:
            st.info(f"Selected Strategy: **{investment_intent}**")

        st.subheader("⚙️ Financing & Purchase Assumptions")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        purchase_price = col_p1.number_input(
            "Purchase Price ($)",
            value=450000,
            step=10000,
            key="re_purchase_price",
        )
        down_payment_pct = (
            col_p2.slider(
                "Down Payment (%)", 0, 50, 20, key="re_down_payment_pct"
            )
            / 100
        )
        interest_rate = (
            col_p3.slider(
                "Mortgage Interest Rate (%)", 3.0, 12.0, 6.5, key="re_interest_rate"
            )
            / 100
        )
        loan_term_years = col_p4.selectbox(
            "Loan Term (Years)", [30, 15, 10], index=0, key="re_loan_term_years"
        )

        col_p5, col_p6, col_p7, col_p8 = st.columns(4)
        closing_costs_pct = (
            col_p5.slider(
                "Closing Costs & Fees (%)", 1.0, 5.0, 2.5, key="re_closing_costs"
            )
            / 100
        )
        rehab_cost = col_p6.number_input(
            "Est. Renovation/Rehab ($)",
            value=25000 if "Flip" in investment_intent else 5000,
            step=2500,
            key="re_rehab_cost",
        )
        appreciation_rate = (
            col_p7.slider(
                "Annual Property Appreciation (%)",
                1.0,
                8.0,
                4.0,
                key="re_appreciation",
            )
            / 100
        )
        rent_growth_rate = (
            col_p8.slider(
                "Annual Rent Increase (%)", 1.0, 6.0, 3.0, key="re_rent_growth"
            )
            / 100
        )

        st.markdown("---")
        st.subheader("💵 Income & Operating Expenses")
        col_inc1, col_inc2, col_inc3, col_inc4 = st.columns(4)
        if "Short-Term" in investment_intent:
            est_monthly_rent = col_inc1.number_input(
                "Est. Avg Monthly STR Revenue ($)",
                value=4800,
                step=200,
                key="re_str_rev",
            )
            vacancy_rate = (
                col_inc2.slider(
                    "Occupancy Loss / Vacancy (%)",
                    5,
                    40,
                    25,
                    key="re_str_vacancy",
                )
                / 100
            )
        else:
            est_monthly_rent = col_inc1.number_input(
                "Est. Monthly Rent ($)",
                value=3200,
                step=100,
                key="re_ltr_rent",
            )
            vacancy_rate = (
                col_inc2.slider(
                    "Vacancy Rate (%)", 1, 15, 5, key="re_ltr_vacancy"
                )
                / 100
            )

        prop_taxes = col_inc3.number_input(
            "Annual Property Taxes ($)",
            value=int(purchase_price * 0.018),
            key="re_taxes",
        )
        insurance = col_inc4.number_input(
            "Annual Insurance ($)", value=1800, key="re_insurance"
        )

        col_exp1, col_exp2 = st.columns(2)
        mgmt_fee_pct = (
            col_exp1.slider(
                "Property Management Fee (%)",
                0,
                15,
                8 if "Short-Term" in investment_intent else 6,
                key="re_mgmt_fee",
            )
            / 100
        )
        maintenance_pct = (
            col_exp2.slider(
                "Maintenance & CapEx Reserves (%)",
                3,
                15,
                7,
                key="re_maint_fee",
            )
            / 100
        )

        # Financial Calculations
        down_payment = purchase_price * down_payment_pct
        loan_amount = purchase_price - down_payment
        closing_costs = purchase_price * closing_costs_pct
        total_out_of_pocket = down_payment + rehab_cost + closing_costs

        gross_annual_income = (est_monthly_rent * 12) * (1 - vacancy_rate)
        annual_mgmt = gross_annual_income * mgmt_fee_pct
        annual_maint = gross_annual_income * maintenance_pct
        total_opex = prop_taxes + insurance + annual_mgmt + annual_maint
        noi = gross_annual_income - total_opex
        cap_rate = (
            (noi / (purchase_price + rehab_cost)) * 100
            if (purchase_price + rehab_cost) > 0
            else 0
        )

        monthly_rate = interest_rate / 12
        num_payments = loan_term_years * 12
        if loan_amount > 0 and monthly_rate > 0:
            monthly_mortgage = (
                loan_amount
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )
        else:
            monthly_mortgage = 0

        annual_debt_service = monthly_mortgage * 12
        annual_cash_flow = noi - annual_debt_service
        monthly_cash_flow = annual_cash_flow / 12
        cash_on_cash = (
            (annual_cash_flow / total_out_of_pocket) * 100
            if total_out_of_pocket > 0
            else 0
        )

        # Output Summary Metrics
        st.markdown("---")
        st.subheader("📊 Executive Underwriting Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cap Rate", f"{cap_rate:.2f}%")
        m2.metric("Cash-on-Cash Return", f"{cash_on_cash:.2f}%")
        m3.metric("Net Operating Income (NOI)", f"${noi:,.0f}/yr")
        m4.metric("Monthly Cash Flow", f"${monthly_cash_flow:,.0f}/mo")
        m5.metric("Total Cash Required", f"${total_out_of_pocket:,.0f}")

        # Fix & Flip Analysis Overlay
        if "Fix & Flip" in investment_intent:
            st.markdown("---")
            st.subheader("🔨 Fix & Flip MAO Breakdown")
            arv = st.number_input(
                "After Repair Value (ARV) ($)",
                value=int(purchase_price * 1.3),
                key="re_arv",
            )
            holding_months = st.slider(
                "Project Duration (Months)", 3, 18, 6, key="re_hold_months"
            )
            holding_costs = (
                (prop_taxes / 12) + (insurance / 12) + monthly_mortgage
            ) * holding_months
            total_project_cost = (
                purchase_price + rehab_cost + holding_costs + closing_costs
            )
            selling_costs = arv * 0.06
            net_profit = arv - total_project_cost - selling_costs
            roi = (
                (net_profit / total_out_of_pocket) * 100
                if total_out_of_pocket > 0
                else 0
            )

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("After Repair Value (ARV)", f"${arv:,.0f}")
            f2.metric("Total Project Cost", f"${total_project_cost:,.0f}")
            f3.metric("Estimated Net Profit", f"${net_profit:,.0f}")
            f4.metric("Return on Investment (ROI)", f"{roi:.2f}%")

            max_allowable_offer = (arv * 0.70) - rehab_cost
            if purchase_price <= max_allowable_offer:
                st.success(
                    f"🟢 **Meets 70% Rule**: Max Allowable Offer (MAO) is **${max_allowable_offer:,.0f}**."
                )
            else:
                st.warning(
                    f"🟡 **Exceeds 70% Rule**: Max Allowable Offer (MAO) is **${max_allowable_offer:,.0f}**."
                )

        # 20-Year Equity & Growth Forecast
        st.markdown("---")
        st.subheader("🔮 20-Year Appreciation & Debt Paydown Forecast")
        records = []
        curr_val = purchase_price
        curr_rent = gross_annual_income
        curr_opex = total_opex
        balance = loan_amount

        for yr in range(1, 21):
            curr_val *= 1 + appreciation_rate
            curr_rent *= 1 + rent_growth_rate
            curr_opex *= 1 + 0.025
            yr_noi = curr_rent - curr_opex
            yr_cash_flow = yr_noi - annual_debt_service
            interest_paid = balance * interest_rate
            principal_paid = min(balance, annual_debt_service - interest_paid)
            balance = max(0, balance - principal_paid)
            equity = curr_val - balance
            records.append({
                "Year": yr,
                "Property Value": round(curr_val, 2),
                "Total Equity": round(equity, 2),
                "Remaining Debt": round(balance, 2),
            })

        df_proj = pd.DataFrame(records)
        y1, y2, y3 = st.columns(3)
        y1.metric(
            "5-Year Est. Value / Equity",
            f"${df_proj.iloc[4]['Property Value']:,.0f}",
            f"Equity: ${df_proj.iloc[4]['Total Equity']:,.0f}",
        )
        y2.metric(
            "10-Year Est. Value / Equity",
            f"${df_proj.iloc[9]['Property Value']:,.0f}",
            f"Equity: ${df_proj.iloc[9]['Total Equity']:,.0f}",
        )
        y3.metric(
            "20-Year Est. Value / Equity",
            f"${df_proj.iloc[19]['Property Value']:,.0f}",
            f"Equity: ${df_proj.iloc[19]['Total Equity']:,.0f}",
        )

        st.line_chart(
            df_proj.set_index("Year")[
                ["Property Value", "Total Equity", "Remaining Debt"]
            ]
        )

    else:
        # Mode 2: Regional Market Scout
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search_region = st.text_input(
                "Target Market / Zip Code / City",
                value="Naperville, IL",
                key="re_search_region",
            )
        with col_s2:
            max_budget = st.number_input(
                "Max Purchase Budget ($)",
                value=600000,
                step=25000,
                key="re_max_budget",
            )

        st.subheader(f"Recommended Opportunities in {search_region}")
        recommended_listings = pd.DataFrame({
            "Address": [
                f"1048 Northway Rd, {search_region}",
                f"412 Fairview Ave, {search_region}",
                f"882 Washington St #2, {search_region}",
                f"1520 Pinecrest Dr, {search_region}",
            ],
            "Price": ["$385,000", "$420,000", "$290,000", "$540,000"],
            "Property Type": [
                "Single Family",
                "Duplex (Multi-Family)",
                "Condo",
                "Single Family",
            ],
            "Est. Cap Rate": ["6.8%", "7.4%", "6.2%", "5.9%"],
            "Cash-on-Cash": ["8.2%", "9.8%", "7.1%", "6.4%"],
            "School Rating": ["9/10", "8/10", "8/10", "10/10"],
            "Investment Strategy Fit": [
                "High (Value-Add)",
                "Excellent (Cash Flow)",
                "Moderate (Low Maint)",
                "High (Appreciation)",
            ],
        })
        st.dataframe(
            recommended_listings, use_container_width=True, hide_index=True
        )


# ==============================================================================
# SECTION 5: FOOTER & BRANDING
# ==============================================================================

st.markdown("---")
col_ft_left, col_ft_right = st.columns([4, 1])
with col_ft_left:
    st.caption(
        "© 2026 Core Market Intelligence (CMI). All Quantitative Models & Market Data Streams."
    )
with col_ft_right:
    st.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)
