import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from groq import Groq
from datetime import timedelta

# ==========================================
# PAGE SETUP & CUSTOM CSS FOR GREAT UI
# ==========================================
st.set_page_config(page_title="AI Gas Forecaster", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI Natural Gas Pricing Intelligence")
st.markdown("Enterprise-grade forecasting powered by **XGBoost** and **Generative AI**.")

# ==========================================
# 1. DATA PIPELINE & MODEL ENGINE
# ==========================================
@st.cache_data
def load_data_and_train_model():
    df = pd.read_csv('Nat_Gas.csv')
    df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
    df = df.sort_values('Dates').reset_index(drop=True)
    
    df['Ordinal'] = df['Dates'].apply(lambda x: x.toordinal())
    df['Month_Sin'] = np.sin(2 * np.pi * df['Dates'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Dates'].dt.month / 12)
    df['Energy_Demand_Index'] = 100 + (df['Dates'].dt.month % 6) * 5
    
    features = ['Ordinal', 'Month_Sin', 'Month_Cos', 'Energy_Demand_Index']
    X = df[features]
    y = df['Prices']
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return df, model, features

df, model, features = load_data_and_train_model()
current_price = df['Prices'].iloc[-1]
max_date = df['Dates'].max()

def predict_future(target_date):
    ordinal = target_date.toordinal()
    m_sin = np.sin(2 * np.pi * target_date.month / 12)
    m_cos = np.cos(2 * np.pi * target_date.month / 12)
    sim_demand = 100 + (target_date.month % 6) * 5
    future_X = pd.DataFrame([[ordinal, m_sin, m_cos, sim_demand]], columns=features)
    return float(model.predict(future_X)[0])

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    target_date = st.date_input("Target Forecast Date:", value=pd.to_datetime("2025-01-15"))
    target_date = pd.to_datetime(target_date)
    forecast_price = predict_future(target_date)
 api_key = None
    
    price_delta = forecast_price - current_price

# ==========================================
# ==========================================
# LIVE COMMODITY FEED SIMULATION
# ==========================================
import time

st.markdown("### 🌐 Live Market Data Stream")
ticker_placeholder = st.empty()

# We cache a baseline session price so it oscillates cleanly rather than jumping wildly
if "live_oil_price" not in st.session_state:
    st.session_state.live_oil_price = 78.50
    st.session_state.oil_history = [78.10, 78.35, 78.20, 78.45, 78.50]

# High-frequency simulation update loop
# Creates a non-blocking fluctuation pattern
price_fluctuation = np.random.normal(0, 0.15)
st.session_state.live_oil_price = round(st.session_state.live_oil_price + price_fluctuation, 2)
st.session_state.oil_history.append(st.session_state.live_oil_price)

# Maintain a rolling window of the last 15 points
if len(st.session_state.oil_history) > 15:
    st.session_state.oil_history.pop(0)

# Build a compact live-updating sparkline
sparkline = go.Figure()
sparkline.add_trace(go.Scatter(
    y=st.session_state.oil_history, 
    mode='lines+markers',
    line=dict(color='#2ca02c' if price_fluctuation >= 0 else '#d62728', width=2),
    marker=dict(size=4)
))
sparkline.update_layout(
    height=80, margin=dict(l=10, r=10, t=0, b=0),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    template="plotly_white", showlegend=False
)

with ticker_placeholder.container():
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        st.metric(
            label="Live WTI Crude Oil (Sim)", 
            value=f"${st.session_state.live_oil_price:.2f}", 
            delta=f"{price_fluctuation:+.2f}"
        )
    with col_t2:
        st.plotly_chart(sparkline, use_container_width=True, config={'displayModeBar': False})

# Forces Streamlit to auto-rerun this specific layout section every 3 seconds
time.sleep(3.0)
st.rerun()
# 3. TABBED UI LAYOUT
# ==========================================
tab1, tab2 = st.tabs(["📊 Market Dashboard", "💬 AI Financial Analyst"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.subheader(f"Forecast Overview: {target_date.strftime('%B %Y')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price (Baseline)", f"${current_price:.2f}")
    col2.metric("AI Forecasted Price", f"${forecast_price:.2f}", delta=f"${price_delta:.2f}")
    col3.metric("Prediction Engine", "XGBoost Regressor")
    
    st.divider()
    
    st.markdown("#### 12-Month Trajectory")
    future_dates = pd.date_range(start=max_date, end=max_date + timedelta(days=365), freq='ME')
    future_prices = [predict_future(d) for d in future_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Dates'], y=df['Prices'], name='Historical', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Forecast', line=dict(color='#ff7f0e', dash='dot', width=3)))
    fig.add_vline(x=max_date, line_width=2, line_dash="dash", line_color="red", annotation_text="Today")
    
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: CONVERSATIONAL AGENT ---
with tab2:
    st.subheader("Chat with the Quantitative Analyst Agent")
    st.caption("Ask about the forecast, market trends, or how the XGBoost model came to its conclusion.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I am your AI Analyst. I see our model predicts Natural Gas will hit **${forecast_price:.2f}** by **{target_date.strftime('%B %Y')}**. How can I help you analyze this?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the analyst a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            active_key = api_key if api_key else st.secrets.get("GROQ_API_KEY")
            
            if active_key:
                try:
                    client = Groq(api_key=active_key)
                    system_context = f"""You are a J.P. Morgan Quantitative Analyst. You are polite, highly analytical, and concise. 
                    Current Gas Price: ${current_price:.2f}. 
                    Target Date selected by user: {target_date.strftime('%B %Y')}. 
                    XGBoost Forecasted Price: ${forecast_price:.2f}.
                    Base your answers on these numbers and general energy market knowledge."""
                    
                    messages = [{"role": "system", "content": system_context}] + st.session_state.messages
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.7
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"API Error: Please check your Groq Key. ({e})")
            else:
                reply = f"*(Simulated Response)* That's a great question about the ${forecast_price:.2f} forecast for {target_date.strftime('%B %Y')}. In a live environment with an API key, I would analyze this utilizing advanced market heuristics. For now, trust the XGBoost model's seasonal mapping!"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
