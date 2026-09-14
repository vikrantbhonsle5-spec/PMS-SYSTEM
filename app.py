import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import io

st.set_page_config(page_title="PMS Stock Filter & Candlestick Analyzer", layout="wide")

st.title("📊 PMS Stock Filter & Candlestick Pattern Analyzer")

# Default Watchlist
DEFAULT_PORTFOLIO = [
    {"Symbol": "RELIANCE.NS", "Company": "Reliance Industries", "Sector": "Energy"},
    {"Symbol": "TCS.NS", "Company": "Tata Consultancy Services", "Sector": "IT"},
    {"Symbol": "INFY.NS", "Company": "Infosys", "Sector": "IT"},
    {"Symbol": "HDFCBANK.NS", "Company": "HDFC Bank", "Sector": "Financials"},
    {"Symbol": "TATAMOTORS.NS", "Company": "Tata Motors", "Sector": "Automobile"},
    {"Symbol": "AAPL", "Company": "Apple Inc.", "Sector": "Technology"},
    {"Symbol": "TSLA", "Company": "Tesla Inc.", "Sector": "Automobile"}
]

@st.cache_data
def get_default_data():
    return pd.DataFrame(DEFAULT_PORTFOLIO)

# Sidebar - Portfolio Loader
st.sidebar.header("1. Watchlist Source")
uploaded_file = st.sidebar.file_uploader("Upload custom PMS.xlsx (Optional)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        pms_df = pd.read_excel(uploaded_file)
        st.sidebar.success("Loaded uploaded Excel!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        pms_df = get_default_data()
else:
    pms_df = get_default_data()

# Parse Tickers
symbol_col = next((col for col in pms_df.columns if col.lower() in ['symbol', 'ticker', 'stock', 'code']), None)
if symbol_col:
    tickers_list = pms_df[symbol_col].dropna().astype(str).str.strip().tolist()
else:
    tickers_list = [d['Symbol'] for d in DEFAULT_PORTFOLIO]

with st.expander("📁 View / Download Current Watchlist", expanded=False):
    st.dataframe(pms_df, use_container_width=True)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pms_df.to_excel(writer, index=False, sheet_name='PMS_Watchlist')
    
    st.download_button(
        label="📥 Download current list as PMS.xlsx",
        data=output.getvalue(),
        file_name="PMS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Candlestick Pattern Logic
def detect_candlestick_patterns(df):
    if df is None or len(df) < 2:
        return "Insufficient Data", "Neutral"
    
    prev, curr = df.iloc[-2], df.iloc[-1]
    c_open, c_close, c_high, c_low = float(curr['Open']), float(curr['Close']), float(curr['High']), float(curr['Low'])
    p_open, p_close = float(prev['Open']), float(prev['Close'])
    
    body = abs(c_close - c_open)
    candle_range = c_high - c_low
    is_bullish = c_close > c_open
    is_bearish = c_close < c_open
    
    if body <= (candle_range * 0.1) and candle_range > 0:
        return "Doji", "Indecision / Reversal Warning"
    elif p_close < p_open and is_bullish and c_close >= p_open and c_open <= p_close:
        return "Bullish Engulfing", "Bullish (Up ↑)"
    elif p_close > p_open and is_bearish and c_close <= p_open and c_open >= p_close:
        return "Bearish Engulfing", "Bearish (Down ↓)"
    elif (c_high - max(c_open, c_close)) < (body * 0.5) and (min(c_open, c_close) - c_low) >= (2 * body):
        return "Hammer / Pinbar", "Bullish (Up ↑)"
    elif (c_high - max(c_open, c_close)) >= (2 * body) and (min(c_open, c_close) - c_low) < (body * 0.5):
        return "Shooting Star", "Bearish (Down ↓)"
    elif is_bullish:
        return "Bullish Candle", "Bullish (Up ↑)"
    elif is_bearish:
        return "Bearish Candle", "Bearish (Down ↓)"
    
    return "No Clear Pattern", "Neutral"

# Fixed Data Fetcher (Handles YFinance Updates)
@st.cache_data(ttl=600, show_spinner=False)
def fetch_resampled_data(ticker):
    try:
        # Download data explicitly without MultiIndex
        data = yf.download(ticker, period="1y", interval="1d", progress=False, multi_level_index=False)
        if data.empty or len(data) == 0:
            return None, None, None
            
        df_d = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_w = df_d.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        df_m = df_d.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        return df_d, df_w, df_m
    except Exception:
        return None, None, None

# App UI Controls
st.sidebar.header("2. Choose Stock")
selected_ticker = st.sidebar.selectbox("Select Ticker from List:", tickers_list)
custom_ticker = st.sidebar.text_input("Or enter custom Ticker (e.g. TATAMOTORS.NS, NVDA):")

ticker_to_run = custom_ticker.strip() if custom_ticker.strip() else selected_ticker

st.divider()

if ticker_to_run:
    st.header(f"📈 Pattern Analysis: {ticker_to_run.upper()}")
    
    with st.spinner("Fetching market data from Yahoo Finance..."):
        df_d, df_w, df_m = fetch_resampled_data(ticker_to_run)
    
    if df_d is None or df_d.empty:
        st.error(f"⚠️ Could not fetch data for '{ticker_to_run}'. Make sure it's a valid symbol on Yahoo Finance (e.g., `RELIANCE.NS` or `AAPL`).")
    else:
        pat_d, sig_d = detect_candlestick_patterns(df_d)
        pat_w, sig_w = detect_candlestick_patterns(df_w)
        pat_m, sig_m = detect_candlestick_patterns(df_m)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("🗓️ Daily Pattern")
            st.info(f"**Pattern:** {pat_d}")
            st.metric("Expected Direction", sig_d)
            st.caption(f"Close: {float(df_d['Close'].iloc[-1]):.2f}")
            
        with c2:
            st.subheader("📅 Weekly Pattern")
            st.info(f"**Pattern:** {pat_w}")
            st.metric("Expected Direction", sig_w)
            st.caption(f"Close: {float(df_w['Close'].iloc[-1]):.2f}")
            
        with c3:
            st.subheader("📆 Monthly Pattern")
            st.info(f"**Pattern:** {pat_m}")
            st.metric("Expected Direction", sig_m)
            st.caption(f"Close: {float(df_m['Close'].iloc[-1]):.2f}")
            
        st.divider()
        
        st.subheader("🕯️ Interactive Price Chart")
        tf_choice = st.radio("Select View Timeframe:", ["Daily", "Weekly", "Monthly"], horizontal=True)
        
        chart_df = df_d if tf_choice == "Daily" else (df_w if tf_choice == "Weekly" else df_m)
        
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'], high=chart_df['High'],
            low=chart_df['Low'], close=chart_df['Close'],
            name=tf_choice
        )])
        
        fig.update_layout(
            title=f"{ticker_to_run.upper()} - {tf_choice} View",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
