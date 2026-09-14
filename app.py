import io
import gzip
import math
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="PMS Stock Filter — NSE / BSE", layout="wide")

# -----------------------------
# PMS scoring rules from PMS.xlsx
# -----------------------------
FUND_WEIGHTS = {
    "Sales Growth": 5,
    "Profit Growth": 5,
    "EPS Growth": 4,
    "ROCE": 4,
    "ROE": 3,
    "Debt/Equity": 3,
    "Operating Cash Flow": 3,
    "Promoter Holding/Pledge": 2,
    "Valuation": 1,
}

TECH_WEIGHTS = {
    "Price > 200 DMA": 2,
    "Price > 50 DMA": 2,
    "50 DMA > 200 DMA": 2,
    "200 DMA rising": 2,
    "Support": 2,
    "Resistance / breakout proximity": 2,
    "Volume": 2,
    "RSI 55-70": 2,
    "Higher high / higher low": 2,
    "Breakout": 2,
}

UPSTOX_NSE_INSTRUMENTS = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_BSE_INSTRUMENTS = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}

def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def pct(x):
    return np.nan if pd.isna(x) else float(x) * 100

def score_positive_growth(v, max_points):
    if pd.isna(v): return 0
    if v >= 20: return max_points
    if v >= 10: return max_points * 0.8
    if v > 0: return max_points * 0.5
    return 0

def score_roce(v):
    if pd.isna(v): return 0
    if v >= 20: return 4
    if v >= 15: return 3.5
    if v >= 10: return 2.5
    if v >= 5: return 1
    return 0

def score_roe(v):
    if pd.isna(v): return 0
    if v >= 18: return 3
    if v >= 15: return 2.7
    if v >= 12: return 2.2
    if v >= 8: return 1
    return 0

def score_de(v):
    if pd.isna(v): return 1.5
    if v <= 0.3: return 3
    if v <= 0.6: return 2.5
    if v <= 1.0: return 1.5
    if v <= 1.5: return 0.75
    return 0

def score_ocf(v):
    if pd.isna(v): return 0
    return 3 if v > 0 else 0

def score_promoter(v):
    if pd.isna(v): return 1
    if v >= 60: return 2
    if v >= 40: return 1.5
    if v >= 25: return 1
    return 0.5

def score_pe(pe):
    if pd.isna(pe) or pe <= 0: return 0.5
    if pe <= 15: return 1
    if pe <= 25: return 0.8
    if pe <= 40: return 0.5
    return 0

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def max_drawdown(close):
    peak = close.cummax()
    dd = close / peak - 1
    return abs(dd.min()) * 100 if len(dd) else np.nan

# -----------------------------
# Candlestick Pattern Engine
# -----------------------------
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

def _download_upstox_instruments(url, exchange):
    r = requests.get(url, headers={
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/gzip, application/octet-stream, */*",
    }, timeout=45)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    data = pd.read_json(io.BytesIO(raw))
    if data.empty:
        return pd.DataFrame()

    data["segment"] = data.get("segment", "").astype(str).str.upper()
    data["instrument_type"] = data.get("instrument_type", "").astype(str).str.upper()
    data = data[(data["segment"].eq(f"{exchange}_EQ")) & (data["instrument_type"].eq("EQ"))].copy()

    if data.empty:
        return pd.DataFrame()

    symbol = data.get("trading_symbol", pd.Series(index=data.index, dtype=str)).astype(str).str.strip().str.upper()
    name = data.get("name", data.get("short_name", symbol)).astype(str).str.strip()
    isin = data.get("isin", pd.Series(index=data.index, dtype=str)).astype(str).str.strip().replace("nan", "")
    token = data.get("exchange_token", pd.Series(index=data.index, dtype=str)).astype(str).str.strip()

    out = pd.DataFrame({
        "symbol": symbol,
        "name": name,
        "exchange": exchange,
        "isin": isin,
        "instrument_key": data.get("instrument_key", "").astype(str),
        "exchange_token": token,
    })
    out = out[(out["symbol"].ne("")) & (out["symbol"].ne("NAN"))].copy()
    if exchange == "NSE":
        out["ticker"] = out["symbol"] + ".NS"
    else:
        out["bse_code"] = out["exchange_token"].str.extract(r"(\d+)")[0]
        out["ticker"] = out["bse_code"] + ".BO"
        out = out[out["bse_code"].notna()]
    return out.drop_duplicates("instrument_key")

def get_nse_universe():
    try:
        return _download_upstox_instruments(UPSTOX_NSE_INSTRUMENTS, "NSE")
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "ticker", "isin", "instrument_key", "exchange_token"])

def get_bse_universe():
    try:
        return _download_upstox_instruments(UPSTOX_BSE_INSTRUMENTS, "BSE")
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "ticker", "isin", "instrument_key", "exchange_token", "bse_code"])

@st.cache_data(ttl=6*60*60, show_spinner=False)
def build_universe():
    nse = get_nse_universe()
    bse = get_bse_universe()
    if nse.empty and bse.empty:
        return pd.DataFrame()
    all_df = pd.concat([nse, bse], ignore_index=True, sort=False)
    all_df["key"] = all_df["isin"].where(all_df["isin"].fillna("").ne(""), all_df["name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True))
    all_df["universe_source"] = "Upstox BOD instrument master"
    all_df["is_nse"] = all_df["exchange"].eq("NSE")
    all_df = all_df.sort_values(["key", "is_nse"], ascending=[True, False])
    return all_df.reset_index(drop=True)

@st.cache_data(ttl=30*60, show_spinner=False)
def download_history(tickers, period="1y"):
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()
    frames = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        try:
            d = yf.download(batch, period=period, interval="1d", auto_adjust=False,
                             progress=False, group_by="ticker", threads=True)
            if d is not None and not d.empty:
                frames.append(d)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)

def extract_ticker_history(data, ticker):
    if data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                x = data[ticker].copy()
            elif ticker in data.columns.get_level_values(1):
                x = data.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            x = data.copy()
            
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
            
        x.columns = [str(c).title() for c in x.columns]
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in x.columns]
        return x[needed].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

def technical_metrics(h):
    if h.empty or len(h) < 100:
        return {}
    close = h["Close"].astype(float)
    vol = h["Volume"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = calc_rsi(close)
    recent = close.iloc[-1]
    rsi_now = float(rsi.iloc[-1]) if not rsi.empty else np.nan
    avgvol20 = vol.rolling(20).mean().iloc[-1]
    vol_ratio = float(vol.iloc[-1] / avgvol20) if avgvol20 else np.nan
    high20 = close.iloc[-21:-1].max() if len(close) >= 21 else close.max()
    low20 = close.iloc[-21:-1].min() if len(close) >= 21 else close.min()
    ret3m = (recent / close.iloc[-64] - 1) * 100 if len(close) >= 64 else np.nan
    ret6m = (recent / close.iloc[-127] - 1) * 100 if len(close) >= 127 else np.nan
    
    d200_val = sma200.iloc[-1] if not pd.isna(sma200.iloc[-1]) else sma50.iloc[-1]
    d200_prev = sma200.iloc[-21] if len(sma200) >= 21 and not pd.isna(sma200.iloc[-21]) else d200_val
    
    return {
        "price": recent,
        "dma50": sma50.iloc[-1],
        "dma200": d200_val,
        "dma200_prev20": d200_prev,
        "rsi": rsi_now,
        "volume_ratio": vol_ratio,
        "high20": high20,
        "low20": low20,
        "ret3m": ret3m,
        "ret6m": ret6m,
        "drawdown": max_drawdown(close.iloc[-252:]) if len(close) >= 252 else np.nan,
        "near_high_pct": (high20 - recent) / high20 * 100 if high20 else np.nan,
    }

def score_technical(m):
    if not m: return 0
    p, d50, d200 = m["price"], m["dma50"], m["dma200"]
    score = 0
    score += 2 if not pd.isna(d200) and p > d200 else 0
    score += 2 if not pd.isna(d50) and p > d50 else 0
    score += 2 if not pd.isna(d50) and not pd.isna(d200) and d50 > d200 else 0
    score += 2 if not pd.isna(m["dma200_prev20"]) and not pd.isna(d200) and m["dma200_prev20"] < d200 else 0
    score += 2 if not pd.isna(m["low20"]) and p > m["low20"] * 1.03 else 0
    score += 2 if not pd.isna(m["near_high_pct"]) and 0 <= m["near_high_pct"] <= 5 else 0
    score += 2 if not pd.isna(m["volume_ratio"]) and m["volume_ratio"] >= 1.2 else (1 if not pd.isna(m["volume_ratio"]) and m["volume_ratio"] >= 0.9 else 0)
    score += 2 if not pd.isna(m["rsi"]) and 55 <= m["rsi"] <= 70 else (1 if not pd.isna(m["rsi"]) and 50 <= m["rsi"] <= 75 else 0)
    score += 2 if not pd.isna(d50) and p > d50 and not pd.isna(m["ret3m"]) and m["ret3m"] > 0 else 0
    score += 2 if not pd.isna(m["near_high_pct"]) and m["near_high_pct"] <= 1.5 and not pd.isna(m["volume_ratio"]) and m["volume_ratio"] >= 1.2 else 0
    return round(min(score, 20), 2)

def fetch_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        out = {
            "price": safe_num(info.get("last_price")),
            "market_cap": safe_num(info.get("market_cap")),
        }
        try:
            full = t.get_info()
        except Exception:
            full = {}
        keys = ["returnOnEquity", "returnOnAssets", "debtToEquity", "trailingPE",
                "priceToBook", "profitMargins", "revenueGrowth", "earningsGrowth",
                "enterpriseToEbitda", "totalRevenue", "operatingCashflow",
                "sharesPercentInsiders", "heldPercentInsiders"]
        for k in keys:
            out[k] = safe_num(full.get(k))
        return out
    except Exception:
        return {}

def fundamental_score(info):
    s = 0
    sales_g = pct(info.get("revenueGrowth"))
    profit_g = pct(info.get("earningsGrowth"))
    roe = pct(info.get("returnOnEquity"))
    de = info.get("debtToEquity")
    if not pd.isna(de): de = de / 100 if de > 10 else de
    s += score_positive_growth(sales_g, 5)
    s += score_positive_growth(profit_g, 5)
    s += score_positive_growth(profit_g, 4)
    s += score_roce(np.nan)
    s += score_roe(roe)
    s += score_de(de)
    ocf = info.get("operatingCashflow")
    s += score_ocf(ocf)
    promoter = pct(info.get("heldPercentInsiders"))
    s += score_promoter(promoter)
    s += score_pe(info.get("trailingPE"))
    return round(min(s, 30), 2)

def label_score(score):
    if score >= 80: return "Strong Buy Candidate"
    if score >= 70: return "Watchlist / Buy on confirmation"
    if score >= 60: return "Wait"
    if score >= 50: return "Weak / Avoid"
    return "Reject"

def risk_score(m):
    if not m: return 0, np.nan, np.nan, np.nan
    price = m["price"]
    sl = max(price * 0.92, m["low20"] * 0.98) if not pd.isna(m["low20"]) else price * 0.92
    risk = price - sl
    target = price + 2 * risk if risk > 0 else np.nan
    score = 5 if risk > 0 and (target - price) / risk >= 2 else 2
    return score, sl, target, (target - price) / risk if risk > 0 else np.nan

def build_rows(universe, history, max_fundamentals, min_mcap_cr, min_sales_growth, min_pat_growth, min_roe, min_turnover_cr):
    rows = []
    candidates = []
    for _, u in universe.iterrows():
        h = extract_ticker_history(history, u["ticker"])
        m = technical_metrics(h)
        if not m:
            continue
        if m["price"] <= 0:
            continue
        turnover = float((h["Close"] * h["Volume"]).tail(20).mean()) if not h.empty else 0
        if turnover < min_turnover_cr * 10_000_000:
            continue
        tech = score_technical(m)
        candidates.append((u, m, turnover, tech, h))
    candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
    candidates = candidates[:max_fundamentals]

    nifty_hist = extract_ticker_history(history, "^NSEI")
    nifty_ret3 = np.nan
    if not nifty_hist.empty and len(nifty_hist) >= 64:
        nifty_ret3 = (nifty_hist["Close"].iloc[-1] / nifty_hist["Close"].iloc[-64] - 1) * 100

    for u, m, turnover, tech, h in candidates:
        info = fetch_info(u["ticker"])
        fscore = fundamental_score(info)
        sales_g = pct(info.get("revenueGrowth"))
        pat_g = pct(info.get("earningsGrowth"))
        roe = pct(info.get("returnOnEquity"))
        mcap_cr = info.get("market_cap") / 1e7 if not pd.isna(info.get("market_cap")) else np.nan

        if not pd.isna(mcap_cr) and mcap_cr < min_mcap_cr:
            continue
        if not pd.isna(sales_g) and sales_g < min_sales_growth:
            continue
        if not pd.isna(pat_g) and pat_g < min_pat_growth:
            continue
        if not pd.isna(roe) and roe < min_roe:
            continue

        momentum = 0
        if not pd.isna(m["ret3m"]) and m["ret3m"] > 0: momentum += 5
        if not pd.isna(m["ret3m"]) and m["ret3m"] >= 10: momentum += 2
        if not pd.isna(nifty_ret3) and not pd.isna(m["ret3m"]) and m["ret3m"] > nifty_ret3: momentum += 5
        if not pd.isna(m["rsi"]) and 55 <= m["rsi"] <= 70: momentum += 3
        momentum = min(momentum, 15)

        sector = 5
        sector += 3 if not pd.isna(m["dma200"]) and m["price"] > m["dma200"] else 0
        sector += 2 if not pd.isna(m["volume_ratio"]) and m["volume_ratio"] >= 1 else 0
        sector = min(sector, 10)

        earnings = 0
        if not pd.isna(pat_g) and pat_g > 0: earnings += 4
        if not pd.isna(sales_g) and sales_g > 0: earnings += 3
        if not pd.isna(info.get("earningsGrowth")) and info.get("earningsGrowth") > 0: earnings += 3
        earnings = min(earnings, 10)

        macro = 5
        macro += 5 if not pd.isna(m["ret3m"]) and m["ret3m"] > 0 else 0
        macro = min(macro, 10)

        rscore, sl, target, rr = risk_score(m)
        total = round(fscore + tech + momentum + sector + earnings + macro + rscore, 2)
        
        # Detect Candlestick Pattern
        c_pattern, c_signal = detect_candlestick_patterns(h)

        rows.append({
            "Rank": 0,
            "Stock": u["symbol"],
            "Ticker": u["ticker"],
            "Company": u["name"],
            "Exchange": u["exchange"],
            "Price": round(m["price"], 2),
            "Candle Pattern": c_pattern,
            "Signal": c_signal,
            "Market Cap (₹ Cr)": round(mcap_cr, 1) if not pd.isna(mcap_cr) else np.nan,
            "PMS Score": total,
            "Decision": label_score(total),
            "Fundamental /30": fscore,
            "Technical /20": tech,
            "Momentum /15": momentum,
            "Sector /10": sector,
            "Earnings /10": earnings,
            "Global /10": macro,
            "Risk /5": rscore,
            "RSI": round(m["rsi"], 1) if not pd.isna(m["rsi"]) else np.nan,
            "3M %": round(m["ret3m"], 2) if not pd.isna(m["ret3m"]) else np.nan,
            "Volume x": round(m["volume_ratio"], 2) if not pd.isna(m["volume_ratio"]) else np.nan,
            "50 DMA": round(m["dma50"], 2) if not pd.isna(m["dma50"]) else np.nan,
            "200 DMA": round(m["dma200"], 2) if not pd.isna(m["dma200"]) else np.nan,
            "Stop Loss": round(sl, 2) if not pd.isna(sl) else np.nan,
            "Target (1:2)": round(target, 2) if not pd.isna(target) else np.nan,
            "R:R": round(rr, 2) if not pd.isna(rr) else np.nan,
            "Avg Turnover ₹/day": round(turnover, 0),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["PMS Score", "Technical /20"], ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out

# -----------------------------
# UI Layout
# -----------------------------
st.title("📈 PMS Stock Filter & Candlestick Analyzer — NSE + BSE")
st.caption("Rule-based screening from PMS.xlsx with Candlestick Pattern Detection.")

with st.sidebar:
    st.header("1. Universe")
    universe = build_universe()
    if universe.empty:
        st.error("Could not download the exchange universe. Check your internet connection.")
        st.stop()
    st.success(f"Universe loaded: {len(universe):,} NSE/BSE equity records")
    exchange_choice = st.selectbox("Exchange universe", ["NSE + BSE", "NSE only", "BSE only"])
    if exchange_choice == "NSE only":
        universe = universe[universe.exchange.eq("NSE")].copy()
    elif exchange_choice == "BSE only":
        universe = universe[universe.exchange.eq("BSE")].copy()

    st.header("2. PMS Filters")
    min_mcap_cr = st.number_input("Minimum market cap (₹ crore)", min_value=0.0, value=1000.0, step=500.0)
    min_sales_growth = st.number_input("Minimum Sales growth %", value=0.0, step=1.0)
    min_pat_growth = st.number_input("Minimum PAT growth %", value=0.0, step=1.0)
    min_roe = st.number_input("Minimum ROE %", value=5.0, step=1.0)
    min_turnover = st.number_input("Minimum average daily turnover (₹ crore)", value=0.1, step=0.1)
    max_fundamentals = st.slider("Maximum stocks for detailed fundamental scoring", 50, 1000, 150, 50)

    st.header("3. Scan")
    run = st.button("🚀 RUN PMS SCAN", type="primary", use_container_width=True)
    refresh = st.button("🔄 Refresh exchange universe", use_container_width=True)

if refresh:
    st.cache_data.clear()
    st.rerun()

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

if run:
    with st.spinner("Downloading price history for NSE/BSE universe and calculating technical signals…"):
        tickers = [t for t in universe["ticker"].dropna().unique().tolist()]
        if "^NSEI" not in tickers:
            tickers.append("^NSEI")
        history = download_history(tickers, period="1y")
        st.session_state.history = history
        
    with st.spinner("Scoring candidates against PMS parameters and Candlesticks…"):
        result = build_rows(universe, history, max_fundamentals, min_mcap_cr, min_sales_growth, min_pat_growth, min_roe, min_turnover)
        st.session_state.results = result

results = st.session_state.results

st.subheader("PMS Ranking & Candlestick Analysis")
if results.empty:
    st.info("Click **RUN PMS SCAN** in the sidebar to scan the exchange universe.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks passing", len(results))
    c2.metric("Strong Buy candidates", int((results["PMS Score"] >= 80).sum()))
    c3.metric("Watchlist", int(((results["PMS Score"] >= 70) & (results["PMS Score"] < 80)).sum()))
    c4.metric("Best PMS score", f"{results['PMS Score'].max():.0f}/100")

    show_cols = ["Rank", "Stock", "Company", "Exchange", "Price", "Candle Pattern", "Signal", "PMS Score", "Decision",
                 "Fundamental /30", "Technical /20", "Momentum /15", "Sector /10", "Earnings /10",
                 "Global /10", "Risk /5", "RSI", "3M %", "Volume x", "Stop Loss", "Target (1:2)"]
    st.dataframe(results[show_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download PMS results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=f"PMS_scan_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📊 Interactive Candlestick Chart & Deep Dive")
    selected_stock = st.selectbox("Select stock to view chart:", results["Stock"].tolist())
    
    if selected_stock:
        selected_row = results[results.Stock.eq(selected_stock)].iloc[0]
        st_ticker = selected_row["Ticker"]
        
        cols = st.columns(7)
        for c, name in zip(cols, ["PMS Score", "Fundamental /30", "Technical /20", "Momentum /15", "Sector /10", "Earnings /10", "Risk /5"]):
            c.metric(name, f"{selected_row[name]:.1f}")

        st.markdown(f"### **{selected_row['Stock']} — {selected_row['Company']}** ({selected_row['Exchange']})")
        st.write(f"**Candle Pattern:** {selected_row['Candle Pattern']} | **Signal:** {selected_row['Signal']} | **Decision:** {selected_row['Decision']}")
        st.write(f"Price ₹{selected_row['Price']:.2f} | Stop ₹{selected_row['Stop Loss']:.2f} | Target ₹{selected_row['Target (1:2)']:.2f} | R:R {selected_row['R:R']:.2f}")

        # Render Candlestick Chart
        chart_h = extract_ticker_history(st.session_state.history, st_ticker)
        if not chart_h.empty:
            tf = st.radio("Timeframe View:", ["Daily", "Weekly", "Monthly"], horizontal=True)
            if tf == "Weekly":
                chart_df = chart_h.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            elif tf == "Monthly":
                chart_df = chart_h.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            else:
                chart_df = chart_h

            fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'], high=chart_df['High'],
                low=chart_df['Low'], close=chart_df['Close'],
                name=tf
            )])
            fig.update_layout(
                title=f"{selected_row['Stock']} - {tf} Candlestick Chart",
                yaxis_title="Price (₹)",
                xaxis_title="Date",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
