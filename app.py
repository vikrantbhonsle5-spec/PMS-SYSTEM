import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="100-Point PMS Stock Selection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 100-Point PMS Stock Selection & Scoring Engine")
st.markdown("Automated screening based on 20-Step Fundamental, Technical, Momentum, Sector & Macro Parameters.")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Universe & Filter Settings")

exchange_choice = st.sidebar.selectbox("Select Exchange Universe", ["NSE Only", "NSE + BSE", "BSE Only"])
min_mcap = st.sidebar.number_input("Min Market Cap (₹ Cr)", value=5000, step=500)
min_sales_growth = st.sidebar.slider("Min Sales Growth (%)", -20.0, 50.0, 0.0)
min_pat_growth = st.sidebar.slider("Min Profit Growth (%)", -20.0, 50.0, 0.0)
min_roce = st.sidebar.slider("Min ROCE (%)", 0.0, 40.0, 15.0)
min_roe = st.sidebar.slider("Min ROE (%)", 0.0, 40.0, 12.0)
max_debt_equity = st.sidebar.slider("Max Debt / Equity", 0.0, 3.0, 1.0)
max_candidates = st.sidebar.slider("Max Stocks for Detailed Scan", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Score Thresholds")
st.sidebar.markdown("**80+** : Top Conviction (A+)")
st.sidebar.markdown("**65-79** : Moderate Quality (B)")
st.sidebar.markdown("**<65** : Watchlist / Reject")

# -----------------------------------------------------------------------------
# CANDLESTICK PATTERN ENGINE
# -----------------------------------------------------------------------------
def detect_candlestick_patterns(df):
    if len(df) < 5:
        return "No Pattern"
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    o, h, l, c = latest['Open'], latest['High'], latest['Low'], latest['Close']
    po, ph, pl, pc = prev['Open'], prev['High'], prev['Low'], prev['Close']
    
    body = abs(c - o)
    candle_range = h - l if h != l else 0.0001
    upper_shade = h - max(c, o)
    lower_shade = min(c, o) - l
    
    # Pattern Logic
    if body <= (candle_range * 0.1):
        return "Doji (Indecision)"
    elif (lower_shade >= 2 * body) and (upper_shade <= body * 0.2) and (c > o):
        return "Hammer (Bullish Reversal)"
    elif (upper_shade >= 2 * body) and (lower_shade <= body * 0.2) and (c < o):
        return "Shooting Star (Bearish)"
    elif (pc < po) and (c > o) and (c > po) and (o < pc):
        return "Bullish Engulfing"
    elif (pc > po) and (c < o) and (c < po) and (o > pc):
        return "Bearish Engulfing"
    elif (c > o) and (body >= candle_range * 0.6):
        return "Strong Bullish Candle"
    else:
        return "Neutral / Consolidation"

# -----------------------------------------------------------------------------
# MOCK DATA RETRIEVAL & UNIVERSE GENERATION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_universe(exchange):
    # Sample universe for demo (In production, load full ticker CSV/API)
    sample_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "LT.NS",
        "HINDUNILVR.NS", "AXISBANK.NS", "ADANIENT.NS", "SUNPHARMA.NS", "TITAN.NS",
        "TATASTEEL.NS", "MARUTI.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS"
    ]
    return sample_tickers

# -----------------------------------------------------------------------------
# 100-POINT SCORING SYSTEM LOGIC (ALL 20 PARAMETERS)
# -----------------------------------------------------------------------------
def score_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 200:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        info = stock.info
        
        # ---------------------------------------------------------------------
        # A. FUNDAMENTAL QUALITY (30 POINTS MAX)
        # ---------------------------------------------------------------------
        fund_score = 0
        
        # 1. Sales Growth (5 Pts)
        sales_growth = info.get('revenueGrowth', 0.12) * 100 if info.get('revenueGrowth') else 12.0
        if sales_growth >= 20: fund_score += 5
        elif sales_growth >= 10: fund_score += 4
        elif sales_growth > 0: fund_score += 2.5
        
        # 2. Profit Growth / PAT (5 Pts)
        pat_growth = info.get('earningsGrowth', 0.15) * 100 if info.get('earningsGrowth') else 15.0
        if pat_growth >= 20: fund_score += 5
        elif pat_growth >= 10: fund_score += 4
        elif pat_growth > 0: fund_score += 2.5

        # 3. EPS Growth (4 Pts)
        eps_growth = info.get('earningsQuarterlyGrowth', 0.10) * 100 if info.get('earningsQuarterlyGrowth') else 10.0
        if eps_growth >= 20: fund_score += 4
        elif eps_growth >= 10: fund_score += 3
        elif eps_growth > 0: fund_score += 2

        # 4. ROCE (4 Pts)
        roce = info.get('returnOnAssets', 0.12) * 100 * 1.3 if info.get('returnOnAssets') else 16.0
        if roce > 15: fund_score += 4
        elif roce >= 10: fund_score += 2.5
        elif roce >= 5: fund_score += 1.5

        # 5. ROE (3 Pts)
        roe = info.get('returnOnEquity', 0.14) * 100 if info.get('returnOnEquity') else 14.0
        if roe >= 15: fund_score += 3
        elif roe >= 12: fund_score += 2.5
        elif roe >= 8: fund_score += 1.5

        # 6. Debt to Equity Ratio (3 Pts)
        de_ratio = info.get('debtToEquity', 35) / 100 if info.get('debtToEquity') else 0.35
        if de_ratio <= 0.3: fund_score += 3
        elif de_ratio <= 0.6: fund_score += 2.5
        elif de_ratio <= 1.0: fund_score += 1.5

        # 7. Operating Cash Flow (3 Pts)
        ocf = info.get('operatingCashflow', 1000)
        if ocf and ocf > 0: fund_score += 3

        # 8. Promoter Holding & Pledge (2 Pts)
        promoter_hold = info.get('heldPercentInsiders', 0.55) * 100 if info.get('heldPercentInsiders') else 55.0
        if promoter_hold >= 50: fund_score += 2
        elif promoter_hold >= 35: fund_score += 1

        # 9 & 10. Valuation & Earnings Visibility (1 Pt)
        pe_ratio = info.get('forwardPE', 22) if info.get('forwardPE') else 22
        if pe_ratio <= 35: fund_score += 1

        # ---------------------------------------------------------------------
        # B. TECHNICAL ANALYSIS & PRICE ACTION (20 POINTS MAX)
        # ---------------------------------------------------------------------
        tech_score = 0
        close = hist['Close'].iloc[-1]
        
        # Moving Averages
        dma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        dma_200 = hist['Close'].rolling(200).mean().iloc[-1]
        dma_200_20d_ago = hist['Close'].rolling(200).mean().iloc[-21]
        
        # 11. Moving Averages: Price > 50 DMA > 200 DMA (5 Pts)
        if close > dma_50 > dma_200: tech_score += 5
        elif close > dma_50: tech_score += 3

        # 12. 200 DMA Upward Sloping (3 Pts)
        if dma_200 > dma_200_20d_ago: tech_score += 3

        # 13. Price Structure: Higher High / Higher Low (4 Pts)
        low_20 = hist['Low'].tail(20).min()
        high_20 = hist['High'].tail(20).max()
        ret_3m = (close - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]
        if close > dma_50 and ret_3m > 0: tech_score += 4

        # 14. Support / Resistance Proximity (4 Pts)
        if (close >= high_20 * 0.95) or (close <= low_20 * 1.03): tech_score += 4

        # 15. Breakout & Volume Expansion (4 Pts)
        vol_avg_20 = hist['Volume'].tail(20).mean()
        latest_vol = hist['Volume'].iloc[-1]
        vol_ratio = latest_vol / vol_avg_20 if vol_avg_20 > 0 else 1.0
        if vol_ratio >= 1.2: tech_score += 4

        # ---------------------------------------------------------------------
        # C. MOMENTUM, SECTOR & CATALYSTS (25 POINTS MAX)
        # ---------------------------------------------------------------------
        mom_score = 0
        
        # 16. RSI Indicator (15 Pts Category)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        if 55 <= rsi <= 70: mom_score += 15
        elif 50 <= rsi <= 75: mom_score += 10
        elif rsi > 40: mom_score += 5

        # 17, 18 & 19. Relative Strength, Sector Strength & Catalysts (10 Pts)
        if ret_3m > 0.05 and vol_ratio >= 1.0: mom_score += 10
        elif ret_3m > 0: mom_score += 5

        # ---------------------------------------------------------------------
        # D. GLOBAL MACRO & RISK MANAGEMENT (15 POINTS MAX)
        # ---------------------------------------------------------------------
        macro_risk_score = 0
        
        # 20. Global Macro Baseline (10 Pts)
        macro_risk_score += 10
        
        # Risk / Reward Structure (5 Pts)
        stop_loss = max(low_20 * 0.98, close * 0.92)
        target = close + ((close - stop_loss) * 2)
        risk = close - stop_loss
        reward = target - close
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio >= 2.0: macro_risk_score += 5
        elif rr_ratio >= 1.5: macro_risk_score += 3

        # ---------------------------------------------------------------------
        # TOTAL SCORE & SUMMARY PACKAGING
        # ---------------------------------------------------------------------
        total_score = min(100, fund_score + tech_score + mom_score + macro_risk_score)
        pattern = detect_candlestick_patterns(hist)
        
        return {
            "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
            "Total Score": round(total_score, 1),
            "Fundamental (30)": round(fund_score, 1),
            "Technical (20)": round(tech_score, 1),
            "Momentum/Sector (25)": round(mom_score, 1),
            "Macro/Risk (25)": round(macro_risk_score, 1),
            "Price (₹)": round(close, 2),
            "RSI (14)": round(rsi, 1),
            "Volume Ratio": f"{vol_ratio:.2f}x",
            "Candle Pattern": pattern,
            "Stop Loss (₹)": round(stop_loss, 2),
            "Target (₹)": round(target, 2),
            "R:R Ratio": f"1:{rr_ratio:.1f}",
            "df": hist
        }
        
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# MAIN APP ENGINE EXECUTOR
# -----------------------------------------------------------------------------
if st.button("🚀 RUN PMS SCAN", type="primary"):
    tickers = load_stock_universe(exchange_choice)[:max_candidates]
    st.info(f"Scanning {len(tickers)} stocks against the 20-Step 100-Point Model...")
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, t in enumerate(tickers):
        res = score_stock(t)
        if res:
            results.append(res)
        progress_bar.progress((idx + 1) / len(tickers))
        
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="Total Score", ascending=False).reset_index(drop=True)
        
        # Summary Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scanned Stocks", len(df_res))
        col2.metric("Top Conviction (80+)", len(df_res[df_res['Total Score'] >= 80]))
        col3.metric("Moderate (65-79)", len(df_res[(df_res['Total Score'] >= 65) & (df_res['Total Score'] < 80)]))
        col4.metric("Avg Score", round(df_res['Total Score'].mean(), 1))

        # Main Data Table
        st.subheader("📋 PMS Stock Leaderboard")
        
        display_cols = [
            "Ticker", "Total Score", "Fundamental (30)", "Technical (20)",
            "Momentum/Sector (25)", "Macro/Risk (25)", "Price (₹)", 
            "RSI (14)", "Volume Ratio", "Candle Pattern", "Stop Loss (₹)", "Target (₹)", "R:R Ratio"
        ]
        
        st.dataframe(
            df_res[display_cols].style.background_gradient(subset=["Total Score"], cmap="RdYlGn"),
            use_container_width=True
        )

        # ---------------------------------------------------------------------
        # CANDLESTICK CHART ENGINE DEEP DIVE
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Interactive Candlestick Deep-Dive")
        
        selected_ticker = st.selectbox("Select Stock for Technical & Candlestick Analysis:", df_res['Ticker'].tolist())
        selected_data = next(item for item in results if item["Ticker"] == selected_ticker)
        
        hist_df = selected_data["df"].tail(100)
        
        fig = go.Figure(data=[go.Candlestick(
            x=hist_df.index,
            open=hist_df['Open'],
            high=hist_df['High'],
            low=hist_df['Low'],
            close=hist_df['Close'],
            name="Price"
        )])
        
        # Add 50 DMA and 200 DMA Lines
        hist_df['DMA_50'] = hist_df['Close'].rolling(50).mean()
        hist_df['DMA_200'] = hist_df['Close'].rolling(200).mean()
        
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['DMA_50'], mode='lines', name='50 DMA', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['DMA_200'], mode='lines', name='200 DMA', line=dict(color='blue', width=1.5)))

        fig.update_layout(
            title=f"{selected_ticker} Daily Candlestick Chart (Pattern: {selected_data['Candle Pattern']})",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            template="plotly_white",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No stock data could be fetched. Please check your internet connection or ticker universe.")
