import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import yfinance as yf
from src.predictor import get_lstm_prediction 

# --- 1. SETTINGS & PREMIUM TECH UI ---
# --- 1. SETTINGS & MOBILE-RESPONSIVE UI ---
st.set_page_config(page_title="Quantitative Finance Engine", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=3840&q=80");
        background-size: cover;
    }

    /* --- MOBILE RESPONSIVENESS FIX --- */
    @media (max-width: 800px) {
        /* Force Metrics to stack vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            padding-bottom: 20px;
        }
        /* Make the SVG Title smaller for mobile screens */
        text { font-size: 22px !important; letter-spacing: 2px !important; }
        
        /* Increase padding for touch targets */
        .stButton button { width: 100% !important; height: 50px; }
    }

    /* Premium Box Styling with 100% Opacity */
    .stInfo, .stSuccess, .stWarning, [data-testid="stMetric"] {
        background-color: rgb(5, 5, 5) !important; 
        color: #00f2ff !important;
        border: 2px solid #00f2ff !important;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.4);
        opacity: 1.0 !important;
        border-radius: 15px;
        padding: 20px;
    }
    
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; font-size: 1.8rem !important; }
    
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(20px);
    }
    .main { background-color: rgba(0, 0, 0, 0.6); backdrop-filter: blur(15px); }
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(page_title="Quantitative Finance Engine", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=3840&q=80");
        background-size: cover;
    }
    /* The "Elite" Box Styling: 100% Opacity + Neon Glow */
    .stInfo, .stSuccess, .stWarning, [data-testid="stMetric"] {
        background-color: rgb(5, 5, 5) !important; 
        color: #00f2ff !important;
        border: 2px solid #00f2ff !important;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.4); /* Neon Glow Effect */
        opacity: 1.0 !important;
        border-radius: 15px;
        padding: 20px;
    }
    /* Metric Label/Value Tuning */
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #888 !important; letter-spacing: 1px; }

    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid #00f2ff;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.6); 
        backdrop-filter: blur(15px);
    }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 0 0 10px #00f2ff; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# Modern Tech Header
st.markdown("""<div style='text-align: center; padding-bottom: 20px;'><svg width='100%' height='60'><text x='50%' y='50%' font-size='32' font-family='Courier New' fill='#00f2ff' text-anchor='middle' letter-spacing='8' filter='drop-shadow(0 0 5px #00f2ff)'>QUANTITATIVE FINANCE ENGINE</text></svg></div>""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period="2y", interval="1d", auto_adjust=True)
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        return data if len(data) > 10 else None
    except: return None

# --- 3. NAVIGATION ---
st.sidebar.markdown("# 🕹️ SYSTEM_CONTROLS")
mode = st.sidebar.radio("CHOOSE_MODE", ["Single Asset Forecast", "Dual Asset Comparison"])
st.sidebar.markdown("---")

# --- 4. MODE 1: SINGLE ASSET ---
if mode == "Single Asset Forecast":
    stock_options = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "CUSTOM"]
    selected = st.sidebar.selectbox("TARGET_ASSET", stock_options)
    ticker = st.sidebar.text_input("MANUAL_TICKER", "SBIN.NS").upper() if selected == "CUSTOM" else selected
    if ".NS" not in ticker and "." not in ticker: ticker += ".NS"

    df = fetch_stock_data(ticker)
    if df is not None:
        close_col = next((c for c in df.columns if c in ['Close', 'Price']), df.columns[0])
        last_price = float(df[close_col].iloc[-1])
        ma50 = df[close_col].rolling(window=50).mean().iloc[-1]
        
        # High-Fidelity Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[close_col], line=dict(color='#00f2ff', width=3), fill='tozeroy', fillcolor='rgba(0, 242, 255, 0.1)'))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)', margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Opaque Glowing Metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label=f"LIVE_PRICE: {ticker}", value=f"₹{last_price:,.2f}")
        with col2:
            if st.button("🚀 INITIATE LSTM_FORECAST"):
                prediction, error = get_lstm_prediction(df.copy(), close_col)
                if error: st.error(error)
                else: st.session_state['forecast'] = prediction
        with col3:
            if 'forecast' in st.session_state:
                st.metric(label="AI_PROJECTION", value=f"₹{st.session_state['forecast']:,.2f}", delta=f"₹{st.session_state['forecast'] - last_price:.2f}")

        # Opaque Analysis Insights
        st.markdown("### 💡 TECHNICAL_INTELLIGENCE")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.info(f"**Architecture:** Stacked LSTM with temporal windowing. Analyzing volatility clusters and momentum drift for **{ticker}**.")
        with s_col2:
            if last_price > ma50:
                st.success(f"**Sentiment:** Bullish. Asset is holding ₹{last_price - ma50:.2f} above the 50-day average. Uptrend is intact.")
            else:
                st.warning(f"**Sentiment:** Bearish. Asset is trading below 50-day support (₹{ma50:,.2f}). Consolidation detected.")

# --- 5. MODE 2: DUAL ASSET ---
else:
    st.sidebar.subheader("PAIR_PARAMETERS")
    t1 = st.sidebar.text_input("BASE_ASSET", "TCS.NS").upper()
    t2 = st.sidebar.text_input("TARGET_ASSET", "INFY.NS").upper()
    
    df1, df2 = fetch_stock_data(t1), fetch_stock_data(t2)
    if df1 is not None and df2 is not None:
        c1, c2 = next((c for c in df1.columns if c in ['Close', 'Price']), df1.columns[0]), next((c for c in df2.columns if c in ['Close', 'Price']), df2.columns[0])
        df1_norm, df2_norm = (df1[c1] / df1[c1].iloc[0]) * 100, (df2[c2] / df2[c2].iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1.index, y=df1_norm, name=t1, line=dict(color='#00f2ff', width=2.5)))
        fig.add_trace(go.Scatter(x=df2.index, y=df2_norm, name=t2, line=dict(color='#ff00ff', width=2.5)))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.4)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Relative Strength:** Normalized to 100 base. Performance Delta: {t1} ({df1_norm.iloc[-1]-100:.1f}%) | {t2} ({df2_norm.iloc[-1]-100:.1f}%).")