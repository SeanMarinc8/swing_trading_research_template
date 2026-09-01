import base64
from datetime import datetime, timedelta
import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import xml.etree.ElementTree as ET
import urllib.request

# ==============================================================================
# SECTION 1: GLOBAL PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="CMI | Institutional Scanner & Predictive Engine",
    layout="wide",
    page_icon="📈",
)

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

st.sidebar.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)
st.sidebar.markdown("---")

col_header_left, col_header_right = st.columns([3, 1])
with col_header_left:
    st.title("📈 Institutional Trading & Predictive Analytics")
    st.caption("Powered by **CMI (Core Market Intelligence)** Quantitative Engine")
with col_header_right:
    st.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: SIDEBAR CONTROLS & CHEAT SHEET
# ==============================================================================
st.sidebar.header("Global Controls")
timeframe_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
timeframe = st.sidebar.selectbox("Analysis Horizon", timeframe_options, index=4)
st.sidebar.markdown("---")

with st.sidebar.expander("📖 Indicator Cheat Sheet (Beginner Friendly)", expanded=False):
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
def compute_all_indicators(df):
    """Calculates SMA, VWAP, RSI, Bollinger Bands, MACD, ATR, and RVOL metrics."""
    df = df.copy()
    
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Overbought (70)"] = 70
    df["Oversold (30)"] = 30
    
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (std_20 * 2)
    df["BB_Lower"] = df["SMA_20"] - (std_20 * 2)
    df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]
    
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
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
    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
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
    buy_target = min(close, latest["BB_Lower"] if pd.notnull(latest["BB_Lower"]) else close)
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

@st.cache_data(ttl=900)
def fetch_multi_source_news(ticker_name):
    """
    Fetches news from Yahoo Finance along with external feeds (WSJ, Dow Jones, Reuters, CNBC, NYT)
    and formats publication dates strictly into (MM/DD/YY).
    """
    articles = []
    try:
        tk = yf.Ticker(ticker_name)
        yf_news = tk.news
        if yf_news:
            for item in yf_news:
                content = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = content.get("title") or item.get("title")
                link = item.get("link") or content.get("link")
                if not link and "clickThroughUrl" in content and content["clickThroughUrl"]:
                    link = content["clickThroughUrl"].get("url")
                
                pub_time = content.get("pubDate") or item.get("providerPublishTime")
                date_str = datetime.now().strftime("%m/%d/%y")
                if pub_time:
                    try:
                        if isinstance(pub_time, (int, float)):
                            dt = datetime.fromtimestamp(pub_time)
                        else:
                            dt = pd.to_datetime(pub_time)
                        date_str = dt.strftime("%m/%d/%y")
                    except Exception:
                        pass
                provider = content.get("provider") or item.get("publisher")
                publisher = provider.get("displayName") if isinstance(provider, dict) else (provider or "Yahoo Finance")
                if title and link:
                    articles.append({
                        "title": title,
                        "link": link,
                        "publisher": publisher,
                        "date": date_str
                    })
    except Exception:
        pass

    rss_sources = [
        ("Wall Street Journal", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        ("CNBC Markets", "https://search.cnbc.com/rs/search/combined/server/settings/rss.jsp?tab=news&id=15839069"),
        ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ]
    for pub_name, feed_url in rss_sources:
        try:
            req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                for item in root.findall('.//item')[:4]:
                    t_title = item.find('title')
                    t_link = item.find('link')
                    t_date = item.find('pubDate')
                    title = t_title.text if t_title is not None else ""
                    link = t_link.text if t_link is not None else ""
                    
                    date_str = datetime.now().strftime("%m/%d/%y")
                    if t_date is not None and t_date.text:
                        try:
                            dt = pd.to_datetime(t_date.text)
                            date_str = dt.strftime("%m/%d/%y")
                        except Exception:
                            pass
                    if title and link:
                        articles.append({
                            "title": title,
                            "link": link,
                            "publisher": pub_name,
                            "date": date_str
                        })
        except Exception:
            pass
    return articles

def render_news_feed(ticker_obj, ticker_name):
    """Renders recent news headlines with publication dates formatted in (MM/DD/YY)."""
    st.markdown("---")
    st.header(f"📰 Recent Market News & Headline Sentiment ({ticker_name})")
    
    articles = fetch_multi_source_news(ticker_name)
    if articles:
        seen_titles = set()
        count = 0
        for art in articles:
            if art["title"] in seen_titles:
                continue
            seen_titles.add(art["title"])
            
            sentiment_tag = analyze_headline_sentiment(art["title"])
            st.markdown(
                f"{sentiment_tag} **[{art['title']}]({art['link']})** — *{art['publisher']}* ({art['date']})"
            )
            count += 1
            if count >= 8:
                break
    else:
        st.info(f"No active news feeds found for **{ticker_name}**.")

def render_plot_with_zoom(df_chart, columns_to_plot, title, y_title, key_prefix):
    """
    Renders Plotly charts with NO MOUSE SCROLL ZOOM (preventing page scroll interference)
    and adds explicit [+] and [-] zoom buttons on the right side.
    """
    if key_prefix not in st.session_state:
        st.session_state[key_prefix] = len(df_chart)
    cur_window = st.session_state[key_prefix]
    col_chart, col_zoom = st.columns([12, 1])
    
    with col_zoom:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("➕", key=f"{key_prefix}_zoom_in", help="Zoom In (Fewer Periods)"):
            st.session_state[key_prefix] = max(10, int(cur_window * 0.75))
            st.rerun()
        if st.button("➖", key=f"{key_prefix}_zoom_out", help="Zoom Out (More Periods)"):
            st.session_state[key_prefix] = min(len(df_chart), int(cur_window * 1.35))
            st.rerun()
    sliced_df = df_chart.tail(st.session_state[key_prefix])
    fig = go.Figure()
    for col in columns_to_plot:
        if col in sliced_df.columns:
            fig.add_trace(go.Scatter(
                x=sliced_df.index,
                y=sliced_df[col],
                mode='lines',
                name=col
            ))
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

def render_trading_strategies_guide():
    """Renders an interactive expander containing proven trading strategies."""
    with st.expander("💡 Easy Proven Trading Strategies (How to use this dashboard)", expanded=False):
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
    """Renders the comprehensive quantitative dashboard with Plotly zoom controls."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    fmt = "{:,.3f}" if asset_type == "Crypto" else "{:,.2f}"
    forecast_df, buy_target, sell_target, stop_loss, daily_vector = generate_predictive_model(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price", f"${fmt.format(latest['Close'])}", f"{fmt.format(latest['Close']-prev['Close'])}")
    c2.metric("VWAP (Inst. Benchmark)", f"${fmt.format(latest['VWAP'])}" if pd.notnull(latest["VWAP"]) else "N/A")
    c3.metric("14-Period RSI", f"{latest['RSI']:.1f}" if pd.notnull(latest["RSI"]) else "N/A")
    c4.metric("Bollinger %B", f"{latest['BB_Percent']:.2f}" if pd.notnull(latest["BB_Percent"]) else "N/A")
    c5.metric("RVOL (Volume Multiplier)", f"{latest['RVOL']:.2f}x" if pd.notnull(latest["RVOL"]) else "N/A")
    c6.metric("14-Period ATR (Daily Range)", f"${fmt.format(latest['ATR'])}" if pd.notnull(latest["ATR"]) else "N/A")
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
        st.success("🟢 **EXECUTIVE ACTION: BUY / ACCUMULATE NOW**\n\nStrong confluence of bullish signals.")
    elif bear_points >= 4:
        st.error("🔴 **EXECUTIVE ACTION: SELL / TAKE PROFITS NOW**\n\nHeavy overhead resistance detected.")
    else:
        st.warning("🟡 **EXECUTIVE ACTION: WAIT / HOLD (NO CLEAR EDGE RIGHT NOW)**\n\nIndicators show a neutral consolidation.")
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
    else:
        entry_target = curr_price
        exit_target = buy_target
        sl_price = curr_price + (1.5 * atr_val)
        risk_pct = max(0.1, abs((sl_price - entry_target) / entry_target) * 100)
        reward_pct = max(0.1, abs((entry_target - exit_target) / entry_target) * 100)
    with st.container():
        st.markdown(f"#### {lean_color} **If You Had to Act: LEAN {lean_direction}**")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Execution Target Price", f"${fmt.format(entry_target)}")
        m_col2.metric("Take-Profit Target", f"${fmt.format(exit_target)}")
        m_col3.metric("Correlating Stop-Loss", f"${fmt.format(sl_price)}")
        m_col4.metric("Trade Risk Percentage", f"{risk_pct:.2f}%", f"Reward: +{reward_pct:.2f}%")
    render_trading_strategies_guide()
    
    render_plot_with_zoom(
        df[["Close", "VWAP", "BB_Upper", "BB_Lower"]].dropna(),
        ["Close", "VWAP", "BB_Upper", "BB_Lower"],
        f"Price Action vs. VWAP & Volatility Bands ({ticker_name})",
        "Price ($)",
        key_prefix=f"{ticker_name}_vwap"
    )
    render_plot_with_zoom(
        df[["RSI", "Overbought (70)", "Oversold (30)"]].dropna(),
        ["RSI", "Overbought (70)", "Oversold (30)"],
        "RSI Momentum & Overbought/Oversold Bounds",
        "RSI Score",
        key_prefix=f"{ticker_name}_rsi"
    )
    st.subheader("MACD Momentum Acceleration")
    macd_df = df[["MACD_Hist"]].dropna().reset_index()
    macd_df["Color"] = np.where(macd_df["MACD_Hist"] >= 0, "Bullish (Green)", "Bearish (Red)")
    macd_chart = (
        alt.Chart(macd_df)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("MACD_Hist:Q", title="MACD Histogram"),
            color=alt.Color(
                "Color:N",
                scale=alt.Scale(domain=["Bullish (Green)", "Bearish (Red)"], range=["#22C55E", "#EF4444"]),
                legend=alt.Legend(title="Momentum"),
            ),
        )
        .properties(height=220)
    )
    st.altair_chart(macd_chart, use_container_width=True)
    
    st.markdown("---")
    st.header("🔮 10-Period Predictive Trajectory & Target Levels")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimal Buy Target", f"${fmt.format(buy_target)}")
    t2.metric("Take Profit Target", f"${fmt.format(sell_target)}")
    t3.metric("Recommended Stop Loss", f"${fmt.format(stop_loss)}")
    t4.metric("Projected Change Per Period", f"${fmt.format(daily_vector)}/period")
    
    render_plot_with_zoom(
        forecast_df,
        ["Predicted Path", "Upper Target (ATR)", "Lower Support (ATR)"],
        "Forecasted Price Path & Confidence Bands",
        "Price ($)",
        key_prefix=f"{ticker_name}_forecast"
    )
    
    st.markdown("---")
    st.header("📋 Financial Statements & Fundamental Overview")
    if asset_type in ["Stock", "Real Estate"]:
        try:
            info = ticker_obj.info
            income_stmt = ticker_obj.financials
            cash_flow = ticker_obj.cashflow
            rev_growth = info.get("revenueGrowth", 0) or 0
            profit_margin = info.get("profitMargins", 0) or 0
            if rev_growth > 0.10 and profit_margin > 0.15:
                health_grade = "GOOD 🟢"
                card_class = "metric-card-good"
                grade_reason = "Strong double-digit top-line growth combined with robust profit margins."
            elif rev_growth > 0 or profit_margin > 0.05:
                health_grade = "AVERAGE 🟡"
                card_class = "metric-card-avg"
                grade_reason = "Moderate fundamental stability with steady margins."
            else:
                health_grade = "BAD 🔴"
                card_class = "metric-card-bad"
                grade_reason = "Declining top-line growth or squeezed margins."
            st.markdown(
                f"""
            <div class="{card_class}">
                <h3 style="margin:0;">Fundamental Rating: <strong>{health_grade}</strong></h3>
                <p style="margin: 5px 0 10px 0;"><strong>Executive Summary:</strong> {grade_reason}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if income_stmt is not None and not income_stmt.empty:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.subheader("Income Statement ($ Millions)")
                    items_to_show = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
                    existing_items = [item for item in items_to_show if item in income_stmt.index]
                    df_inc = income_stmt.loc[existing_items] / 1e6
                    df_inc.columns = [col.strftime("%Y") if hasattr(col, "strftime") else col for col in df_inc.columns]
                    st.dataframe(df_inc.style.format("${:,.1f}M"), use_container_width=True)
                with col_f2:
                    st.subheader("Cash Flow Statement ($ Millions)")
                    cf_items = ["Operating Cash Flow", "Capital Expenditures", "Free Cash Flow"]
                    existing_cf = [item for item in cf_items if item in cash_flow.index]
                    if existing_cf:
                        df_cf = cash_flow.loc[existing_cf] / 1e6
                        df_cf.columns = [col.strftime("%Y") if hasattr(col, "strftime") else col for col in df_cf.columns]
                        st.dataframe(df_cf.style.format("${:,.1f}M"), use_container_width=True)
        except Exception:
            st.info("Financial statements not available for this ticker.")
    
    render_news_feed(ticker_obj, ticker_name)

# ==============================================================================
# SECTION 4: MAIN INTERFACE & CONTROLLER (TABBED ASSET CLASSES)
# ==============================================================================
asset_type = st.sidebar.radio("Asset Class", ["Stock", "Crypto", "Real Estate"])

recommended_stocks = ["NVDA", "VOO", "RUM"]
recommended_cryptos = ["BTC-USD", "ETH-USD", "SOL-USD"]
recommended_reits = ["O", "PLD", "AMT", "SPG", "EQIX"]

if asset_type == "Stock":
    options = recommended_stocks
    custom_placeholder = "e.g. AAPL, MSFT, TSLA"
elif asset_type == "Crypto":
    options = recommended_cryptos
    custom_placeholder = "e.g. BTC-USD, ETH-USD, SOL-USD"
else: # Real Estate
    options = recommended_reits
    custom_placeholder = "e.g. O, PLD, AMT, SPG"

st.sidebar.markdown("### Select or Custom Search")
ticker_selection = st.sidebar.selectbox("Recommended Options", options)
custom_ticker = st.sidebar.text_input(f"Or enter Custom Ticker ({custom_placeholder})", "").strip().upper()

# Primary ticker selection
ticker_name = custom_ticker if custom_ticker else ticker_selection

if ticker_name:
    try:
        ticker_obj = yf.Ticker(ticker_name)
        hist_df = ticker_obj.history(period=timeframe)
        
        if hist_df.empty:
            st.error(f"No market data found for ticker **'{ticker_name}'**. Please verify the symbol.")
        else:
            processed_df = compute_all_indicators(hist_df)
            render_full_dashboard(processed_df, ticker_name, asset_type, ticker_obj)
    except Exception as e:
        st.error(f"Error retrieving data for **{ticker_name}**: {str(e)}")
else:
    st.info("Please select or enter a ticker symbol to begin analysis.")
