import streamlit as st
import pandas as pd
import requests
import numpy as np
import html
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import time

# ONLY ADD THIS IMPORT AT TOP
from datetime import datetime

st.set_page_config(
    page_title="PowerPulse", layout="wide", initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown(
    """
    <style>
    /* Overall theme - Purple & Lavender Aesthetic */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #A78BFA;
        --accent-color: #DDD6FE;
        --light-bg: #FAF8FF;
        --card-bg: #FFFFFF;
        --text-dark: #2D1B4E;
        --text-light: #6D28D9;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #FAF8FF 0%, #F3EDFF 100%);
    }
    
    /* Header styling */
    .header-main {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 2px 8px rgba(124, 58, 237, 0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F5FF 100%);
        border-radius: 20px;
        padding: 25px;
        border: 3px solid #C4B5FD;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.25);
        border-color: #7C3AED;
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 900;
        color: #FFFFFF;
        background: linear-gradient(90deg, #6D28D9 0%, #8B5CF6 55%, #A78BFA 100%);
        border: 1px solid #C4B5FD;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 30px 0 20px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 18px rgba(124, 58, 237, 0.16);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.18);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #6D28D9 0%, #7C3AED 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        border: 2px solid #C4B5FD;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.4);
        margin: 10px 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs and expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
        border-radius: 8px;
        color: white !important;
    }

    [data-testid="stExpander"] details summary {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #C4B5FD !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 8px 18px rgba(124, 58, 237, 0.16) !important;
    }

    [data-testid="stExpander"] details summary:hover {
        background: linear-gradient(90deg, #8B5CF6 0%, #C4B5FD 100%) !important;
        box-shadow: 0 12px 24px rgba(124, 58, 237, 0.22) !important;
    }

    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    /* Divider */
    hr {
        border: 1px solid #E9D5FF;
        margin: 30px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        padding: 12px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 15px rgba(124, 58, 237, 0.2);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.35);
    }
    
    /* Caption styling */
    .caption-text {
        color: #6D28D9;
        font-style: italic;
        font-size: 14px;
        margin: 15px 0;
        font-weight: 500;
    }
    
    /* Table and DataFrame styling */
    .stDataFrame {
        background: #FFFFFF !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: #FFFFFF !important;
        border-collapse: collapse !important;
    }
    
    .stDataFrame thead {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
    }
    
    .stDataFrame thead th {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: 1px solid #C4B5FD !important;
        padding: 12px !important;
        text-align: center !important;
    }
    
    .stDataFrame tbody td {
        background: #F8F6FF !important;
        color: #2D1B4E !important;
        border: 1px solid #E9D5FF !important;
        padding: 10px !important;
        font-weight: 500 !important;
        text-align: center !important;
    }
    
    .stDataFrame tbody tr:hover td {
        background: #EDE9FE !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) td {
        background: #FFFFFF !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .streamlit-expanderContent {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] details > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

        .table-title {
            color: #FFFFFF;
            background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 0.4px;
            margin: 10px 0 14px 0;
            box-shadow: 0 8px 20px rgba(124, 58, 237, 0.20);
        }

        .table-card {
            background: #FFFFFF;
            border: 1px solid #D8B4FE;
            border-radius: 14px;
            padding: 10px;
            box-shadow: 0 10px 25px rgba(124, 58, 237, 0.10);
            width: 100%;
            min-width: 0;
            height: 520px;
            overflow: auto;
        }

        .table-card table {
            width: 100% !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F5F3FF 0%, #EDE9FE 100%) !important;
            color: #2D1B4E !important;
            border-right: 1px solid #DDD6FE !important;
        }

        [data-testid="stSidebar"] * {
            color: #2D1B4E !important;
        }

        [data-testid="stSidebarNav"] {
            background: transparent !important;
        }

        [data-testid="stSidebarNav"] a {
            background: #FFFFFF !important;
            border: 1px solid #C4B5FD !important;
            border-radius: 12px !important;
            margin-bottom: 8px !important;
            box-shadow: 0 6px 14px rgba(124, 58, 237, 0.10) !important;
        }

        [data-testid="stSidebarNav"] a:hover {
            background: #EDE9FE !important;
            border-color: #A78BFA !important;
            box-shadow: 0 10px 20px rgba(124, 58, 237, 0.16) !important;
        }

        [data-testid="stSidebarNav"] a span,
        [data-testid="stSidebarNav"] a p {
            color: #2D1B4E !important;
            font-weight: 800 !important;
        }

        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h4,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #2D1B4E !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"],
        [data-testid="stSidebar"] .stButton > button {
            border: 1px solid #C4B5FD !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            background: #FFFFFF !important;
            transition: all 0.25s ease !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.06) !important;
            color-scheme: light !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
            color: #2D1B4E !important;
            caret-color: #7C3AED !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {
            fill: #7C3AED !important;
        }

        [data-testid="stSidebar"] [data-baseweb="popover"] {
            background: #FFFFFF !important;
            border: 1px solid #C4B5FD !important;
            border-radius: 14px !important;
            box-shadow: 0 16px 34px rgba(124, 58, 237, 0.22) !important;
        }

        [data-testid="stSidebar"] [role="listbox"] {
            background: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [role="option"] {
            color: #2D1B4E !important;
            background: #FFFFFF !important;
            padding: 10px 12px !important;
        }

        [data-testid="stSidebar"] [role="option"]:hover,
        [data-testid="stSidebar"] [role="option"][aria-selected="true"] {
            background: #EDE9FE !important;
            color: #4C1D95 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"] {
            color: #2D1B4E !important;
            font-weight: 700 !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"] {
            background: #FFFFFF !important;
            border: 1px solid #C4B5FD !important;
            border-radius: 14px !important;
            box-shadow: 0 16px 34px rgba(124, 58, 237, 0.22) !important;
            z-index: 9999 !important;
        }

        [data-baseweb="popover"] *,
        [data-baseweb="menu"] * {
            color: #2D1B4E !important;
            background-color: #FFFFFF !important;
        }

        [data-baseweb="menu"] [role="option"],
        [role="listbox"] [role="option"] {
            color: #2D1B4E !important;
            background: #FFFFFF !important;
            padding: 10px 12px !important;
        }

        [data-baseweb="menu"] [data-baseweb="menu-item"],
        [role="listbox"] [data-baseweb="menu-item"] {
            color: #2D1B4E !important;
            background: #FFFFFF !important;
            padding: 10px 12px !important;
        }

        [data-baseweb="menu"] [role="option"]:hover,
        [data-baseweb="menu"] [role="option"][aria-selected="true"],
        [data-baseweb="menu"] [data-baseweb="menu-item"]:hover,
        [data-baseweb="menu"] [data-baseweb="menu-item"][aria-selected="true"],
        [role="listbox"] [role="option"]:hover,
        [role="listbox"] [role="option"][aria-selected="true"] {
            background: #EDE9FE !important;
            color: #4C1D95 !important;
        }

        [data-baseweb="select"] [class*="singleValue"],
        [data-baseweb="select"] [class*="placeholder"] {
            color: #2D1B4E !important;
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div:hover {
            background: #E0E7FF !important;
            border-color: #8B5CF6 !important;
            box-shadow: 0 10px 20px rgba(139, 92, 246, 0.22) !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div[aria-expanded="true"] {
            background: #EDE9FE !important;
            border-color: #8B5CF6 !important;
            box-shadow: 0 10px 20px rgba(139, 92, 246, 0.24) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"]:hover {
            filter: drop-shadow(0 0 8px rgba(167, 139, 250, 0.35));
        }

        /* Minimal straight slider: use native range styles for consistent thin bar */
        [data-testid="stSidebar"] input[type="range"] {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 100% !important;
            height: 8px !important;
            background: linear-gradient(90deg, #EDE9FE 0%, #F5F3FF 100%) !important;
            border-radius: 999px !important;
            outline: none !important;
            padding: 0 !important;
            margin: 6px 0 2px 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 14px !important;
            height: 14px !important;
            border-radius: 50% !important;
            background: #7C3AED !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 4px 10px rgba(124, 58, 237, 0.25) !important;
            margin-top: -3px !important;
        }

        [data-testid="stSidebar"] input[type="range"]::-moz-range-thumb {
            width: 14px !important;
            height: 14px !important;
            border-radius: 50% !important;
            background: #7C3AED !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 4px 10px rgba(124, 58, 237, 0.25) !important;
        }

        [data-testid="stSidebar"] input[type="range"]::-webkit-slider-runnable-track {
            height: 8px !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, #EDE9FE 0%, #F8F6FF 100%) !important;
            border: 1px solid #E9D5FF !important;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #FFFFFF !important;
            border: 1px solid #DDD6FE !important;
            border-radius: 14px !important;
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.08) !important;
            transition: all 0.25s ease !important;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"]:hover {
            background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%) !important;
            border-color: #A78BFA !important;
            box-shadow: 0 10px 24px rgba(124, 58, 237, 0.16) !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%) !important;
            color: white !important;
            box-shadow: 0 6px 15px rgba(124, 58, 237, 0.18) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(90deg, #8B5CF6 0%, #C4B5FD 100%) !important;
            box-shadow: 0 12px 24px rgba(124, 58, 237, 0.28) !important;
            transform: translateY(-1px) scale(1.02) !important;
        }

        [data-testid="stSidebar"] [data-testid="stInfoBox"] {
            background: #FFFFFF !important;
            border: 1px solid #D8B4FE !important;
        }

        [data-testid="stSidebar"] [data-testid="stInfoBox"] p,
        [data-testid="stSidebar"] [data-testid="stInfoBox"] div,
        [data-testid="stSidebar"] [data-testid="stInfoBox"] span {
            color: #2D1B4E !important;
        }

        .sidebar-panel {
            background: linear-gradient(180deg, #FFFFFF 0%, #FAF5FF 100%);
            border: 1px solid #DDD6FE;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 16px 32px rgba(124, 58, 237, 0.10);
            margin: 0 0 18px 0;
        }

        .sidebar-panel-title {
            background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
            color: #FFFFFF;
            border-radius: 14px;
            padding: 10px 12px;
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.3px;
            margin-bottom: 14px;
            text-align: center;
            box-shadow: 0 8px 18px rgba(124, 58, 237, 0.18);
        }

        .sidebar-section-title {
            color: #4C1D95;
            font-size: 13px;
            font-weight: 800;
            margin: 12px 0 8px 0;
        }

        .forecast-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 100%);
            border: 1px solid #DDD6FE;
            border-radius: 16px;
            padding: 12px 12px 10px 12px;
            margin-top: 6px;
            box-shadow: 0 10px 20px rgba(124, 58, 237, 0.08);
        }

        .forecast-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .forecast-chip {
            background: #EDE9FE;
            color: #4C1D95;
            border: 1px solid #C4B5FD;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.3px;
        }

        .forecast-value {
            background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
            color: #FFFFFF;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 12px;
            font-weight: 800;
            box-shadow: 0 6px 14px rgba(124, 58, 237, 0.18);
        }

        .forecast-range-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #7C3AED;
            font-size: 11px;
            font-weight: 800;
            margin: 0 2px 6px 2px;
        }

        .sidebar-note {
            color: #6D28D9;
            font-size: 12px;
            font-weight: 500;
            margin-top: 8px;
        }

        .sidebar-slider-title {
            color: #2D1B4E;
            font-size: 14px;
            font-weight: 700;
            margin: 8px 0 6px 0;
        }

        .sidebar-slider-range {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #7C3AED;
            font-size: 12px;
            font-weight: 700;
            margin: 0 2px 2px 2px;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] {
            margin-top: 0.25rem !important;
        }
    
    /* Metric value styling */
    .metric-value {
        color: #7C3AED;
        font-weight: bold;
        font-size: 28px;
    }
    
    /* Text elements */
    p {
        color: #2D1B4E;
    }
    
    div {
        color: #2D1B4E;
    }
    
    /* Chart text */
    .stPlotlyChart text {
        fill: #2D1B4E !important;
    }

    .footer-card {
        background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        margin-top: 40px;
        border: 1px solid #C4B5FD;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.10);
    }

    .footer-title {
        color: #2D1B4E;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }

    .footer-subtitle {
        color: #7C3AED;
        font-size: 12px;
        margin-top: 8px;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== ANIMATED HEADER ====================
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='
            background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 56px;
            font-weight: 900;
            margin: 0;
            text-shadow: 0 2px 10px rgba(124, 58, 237, 0.15);
        '>⚡ PowerPulse</h1>
        <p style='
            color: #7C3AED;
            font-size: 18px;
            margin: 10px 0;
            font-weight: 600;
            letter-spacing: 1px;
        '>Energy Monitoring & Prediction Dashboard</p>
    </div>
""",
    unsafe_allow_html=True,
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-panel-title'>🎛️ Control Panel</div>",
        unsafe_allow_html=True,
    )

    cities = {
        "🟦 Bangalore": (12.97, 77.59),
        "🟥 Mumbai": (19.08, 72.88),
        "🟨 Delhi": (28.61, 77.23),
        "🟩 Kolkata": (22.57, 88.36),
    }

    st.markdown(
        "<div class='sidebar-section-title'>📍 Select City</div>",
        unsafe_allow_html=True,
    )
    city_label = st.selectbox("Select City", list(cities.keys()), key="city_select")
    city = city_label.split()[-1]  # Extract city name
    lat, lon = cities[city_label]

    st.markdown(
        "<div class='sidebar-section-title'>🔮 Forecast Hours</div>",
        unsafe_allow_html=True,
    )
    # Minimal straight slider bar for Forecast Hours
    forecast_hours = st.slider(
        "Forecast Hours",
        12,
        48,
        24,
        step=6,
        label_visibility="collapsed",
        key="forecast_hours",
    )
    st.markdown(
        f"<div class='sidebar-slider-range'><span>12h</span><strong style='color:#4C1D95'>{forecast_hours}h</strong><span>48h</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sidebar-section-title'>📊 Dashboard Info</div>",
        unsafe_allow_html=True,
    )
    st.info("""
    ✨ **Real-time Energy Tracking**
    - Live consumption data
    - AI-powered forecasts
    - Trend analysis
    """)

    st.markdown(
        "<div class='sidebar-note'>Choose a city and forecast window, then refresh the data.</div>",
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.toast("✅ Data refreshed!")
        st.rerun()


# closing div
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=300)
def fetch_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m"

    for _ in range(3):
        try:
            res = requests.get(url, timeout=10)
            data = res.json()

            df = pd.DataFrame(
                {
                    "Datetime": pd.to_datetime(data["hourly"]["time"]),
                    "Temperature": data["hourly"]["temperature_2m"],
                }
            )
            return df
        except:
            time.sleep(2)

    return pd.DataFrame()


def render_energy_table(
    df,
    title,
    highlight_max_color="#EDE9FE",
    highlight_min_color="#F5F3FF",
    mark_peak_time=None,
    mark_low_time=None,
):
    # Reset index to ensure Datetime is a column
    df_display = (
        df.reset_index(drop=True)
        if not isinstance(df.index, pd.RangeIndex)
        else df.copy()
    )
    if (
        "Datetime" not in df_display.columns
        and hasattr(df, "index")
        and isinstance(df.index, pd.DatetimeIndex)
    ):
        df_display = df.reset_index()

    if "Datetime" in df_display.columns:
        df_display["Datetime"] = pd.to_datetime(df_display["Datetime"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    if "Datetime" in df_display.columns:

        def build_tag(row):
            row_time = pd.to_datetime(row["Datetime"]).floor("H")
            if mark_peak_time is not None and row_time == pd.to_datetime(
                mark_peak_time
            ).floor("H"):
                return "🔥 Peak"
            if mark_low_time is not None and row_time == pd.to_datetime(
                mark_low_time
            ).floor("H"):
                return "❄️ Low"
            return ""

        df_display["Tag"] = df_display.apply(build_tag, axis=1)

    headers = list(df_display.columns)
    table_rows = []

    for _, row in df_display.iterrows():
        row_style = "background-color: #F8F6FF; color: #2D1B4E;"
        cell_style = (
            "border: 1px solid #E9D5FF; padding: 10px; text-align: center; "
            "background-color: #F8F6FF; color: #2D1B4E;"
        )
        row_time = None
        if "Datetime" in row.index:
            try:
                row_time = pd.to_datetime(row["Datetime"]).floor("H")
            except Exception:
                row_time = None

        if row_time is not None:
            if mark_peak_time is not None and row_time == pd.to_datetime(
                mark_peak_time
            ).floor("H"):
                row_style = (
                    "background-color: #FFB3B3; color: #7F1D1D; font-weight: 700;"
                )
                cell_style = (
                    "border: 1px solid #E9D5FF; padding: 10px; text-align: center; "
                    "background-color: #FFB3B3; color: #7F1D1D; font-weight: 700;"
                )
            elif mark_low_time is not None and row_time == pd.to_datetime(
                mark_low_time
            ).floor("H"):
                row_style = (
                    "background-color: #A7F3D0; color: #14532D; font-weight: 700;"
                )
                cell_style = (
                    "border: 1px solid #E9D5FF; padding: 10px; text-align: center; "
                    "background-color: #A7F3D0; color: #14532D; font-weight: 700;"
                )

        cells = []
        for value in row.tolist():
            cells.append(
                f"<td style='{cell_style}'>{html.escape(str(value))}</td>"
            )
        table_rows.append(f"<tr style='{row_style}'>{''.join(cells)}</tr>")

    header_html = "".join(
        f"<th style='background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%); color: white; font-weight: 700; border: 1px solid #C4B5FD; padding: 12px; text-align: center;'>{html.escape(str(header))}</th>"
        for header in headers
    )

    table_html = f"""
    <table style='border-collapse: collapse; width: 100%; background: #FFFFFF;'>
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    """

    st.markdown(
        f"<div class='table-title'>{title}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='table-card'>{table_html}</div>", unsafe_allow_html=True)


df = fetch_data(lat, lon)

if df.empty:
    st.error("Unable to fetch data")
    st.stop()

df.set_index("Datetime", inplace=True)

# -------------------------------
# PROCESS DATA
# -------------------------------
now = pd.Timestamp.now()
df_real = df[df.index <= now].copy()

df_real["Energy"] = 50 + df_real["Temperature"] * 2
df_real["Energy"] += np.random.normal(0, 1, len(df_real))
st.session_state.df_real = df_real
# ==================== TOP SUMMARY (METRICS CARDS) ====================
st.markdown(
    f'<div class="section-header">📍 {city} - Live Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

current_usage = df_real["Energy"].iloc[-1]
avg_usage = df_real["Energy"].mean()
peak_usage = df_real["Energy"].max()
min_usage = df_real["Energy"].min()

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <div style="font-size: 24px; color: #7C3AED;">⚡</div>
            <div style="color: #6D28D9; font-size: 12px; margin-top: 5px; font-weight: 600;">CURRENT</div>
            <div style="color: #2D1B4E; font-size: 28px; font-weight: bold; margin-top: 10px;">{current_usage:.1f}</div>
            <div style="color: #6D28D9; font-size: 12px;">kWh</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <div style="font-size: 24px; color: #A78BFA;">📊</div>
            <div style="color: #6D28D9; font-size: 12px; margin-top: 5px; font-weight: 600;">AVERAGE</div>
            <div style="color: #2D1B4E; font-size: 28px; font-weight: bold; margin-top: 10px;">{avg_usage:.1f}</div>
            <div style="color: #6D28D9; font-size: 12px;">kWh</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <div style="font-size: 24px; color: #7C3AED;">🔥</div>
            <div style="color: #6D28D9; font-size: 12px; margin-top: 5px; font-weight: 600;">PEAK</div>
            <div style="color: #2D1B4E; font-size: 28px; font-weight: bold; margin-top: 10px;">{peak_usage:.1f}</div>
            <div style="color: #6D28D9; font-size: 12px;">kWh</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <div style="font-size: 24px; color: #A78BFA;">❄️</div>
            <div style="color: #6D28D9; font-size: 12px; margin-top: 5px; font-weight: 600;">MINIMUM</div>
            <div style="color: #2D1B4E; font-size: 28px; font-weight: bold; margin-top: 10px;">{min_usage:.1f}</div>
            <div style="color: #6D28D9; font-size: 12px;">kWh</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==================== ENERGY TREND ====================
st.markdown(
    '<div class="section-header">📈 Energy Trend (Recent Hours)</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_real.tail(48).index,
        y=df_real.tail(48)["Energy"],
        mode="lines+markers",
        name="Energy Usage",
        line=dict(color="#7C3AED", width=4),
        marker=dict(size=7, color="#A78BFA", line=dict(color="#FFFFFF", width=1)),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.12)",
        hovertemplate=(
            "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
            "Historical: %{y:.1f} kWh"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    template="plotly",
    height=450,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Time",
    yaxis_title="Energy Consumption (kWh)",
    hovermode="x unified",
    hoverdistance=10,
    spikedistance=10,
    plot_bgcolor="rgba(255, 255, 255, 1)",
    paper_bgcolor="rgba(255, 255, 255, 1)",
    font=dict(color="#2D1B4E", size=12, family="Arial"),
    xaxis=dict(
        showgrid=True, gridwidth=1, gridcolor="rgba(124, 58, 237, 0.15)", zeroline=False
    ),
    yaxis=dict(
        showgrid=True, gridwidth=1, gridcolor="rgba(124, 58, 237, 0.15)", zeroline=False
    ),
    hoverlabel=dict(
        bgcolor="#EDE9FE",
        font=dict(color="#2D1B4E", size=13, family="Arial"),
        bordercolor="#C4B5FD",
    ),
)

st.plotly_chart(fig, use_container_width=True)
st.markdown(
    "<p class='caption-text'>📊 Real-time energy consumption trend for the last 48 hours</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ==================== FORECAST ====================
# Forecast UI moved to the Forecast & Optimization page

# ==================== KEY INSIGHTS ====================
st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

peak_time = df_real["Energy"].idxmax()
low_time = df_real["Energy"].idxmin()
avg_value = df_real["Energy"].mean()

# Calculate energy stats
total_energy = df_real["Energy"].sum()
energy_variance = df_real["Energy"].std()
if "forecast_df" in st.session_state:
    try:
        forecast_avg = st.session_state["forecast_df"]["Energy"].mean()
    except Exception:
        forecast_avg = float("nan")
else:
    forecast_avg = float("nan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
    <div class="info-box">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; height: 160px;">
            <div style="font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🔥</div>
            <div style="font-size: 12px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2); letter-spacing: 0.5px;">PEAK USAGE</div>
            <div style="font-size: 40px; font-weight: 900; color: #FFFFFF; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">{df_real["Energy"].max():.1f}</div>
            <div style="font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Highest: {peak_time.strftime('%H:%M')}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="info-box">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; height: 160px;">
            <div style="font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📊</div>
            <div style="font-size: 12px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2); letter-spacing: 0.5px;">AVERAGE USAGE</div>
            <div style="font-size: 40px; font-weight: 900; color: #FFFFFF; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">{avg_value:.1f}</div>
            <div style="font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Variation: ±{energy_variance:.1f}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="info-box">
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; height: 160px;">
            <div style="font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">❄️</div>
            <div style="font-size: 12px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2); letter-spacing: 0.5px;">LOWEST USAGE</div>
            <div style="font-size: 40px; font-weight: 900; color: #FFFFFF; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">{df_real["Energy"].min():.1f}</div>
            <div style="font-size: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Lowest: {low_time.strftime('%H:%M')}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==================== ADDITIONAL DATA ====================
with st.expander("📊 View Detailed Data & Analytics"):
    left_col, right_col = st.columns([1, 1], gap="large")

    recent_energy_table = df_real.tail(20).copy()
    recent_peak_time = recent_energy_table["Energy"].idxmax()
    recent_low_time = recent_energy_table["Energy"].idxmin()

    with left_col:
        render_energy_table(
            recent_energy_table.reset_index(),
            "Recent Energy Consumption (Last 20 hours)",
            mark_peak_time=recent_peak_time,
            mark_low_time=recent_low_time,
        )

        recent_energy_table_display = recent_energy_table.reset_index()
        recent_energy_table_display["Tag"] = ""
        recent_energy_table_display.loc[
            recent_energy_table_display["Datetime"].dt.floor("H")
            == pd.Timestamp(recent_peak_time).floor("H"),
            "Tag",
        ] = "🔥 Peak"
        recent_energy_table_display.loc[
            recent_energy_table_display["Datetime"].dt.floor("H")
            == pd.Timestamp(recent_low_time).floor("H"),
            "Tag",
        ] = "❄️ Low"

        # Color legend
        st.markdown(
            """
            <div style='display: flex; gap: 20px; margin-top: 16px; padding: 12px; background: #F5F3FF; border-radius: 10px; border: 1px solid #DDD6FE;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 24px; height: 24px; background-color: #FF6B6B; border: 1px solid #E63946; border-radius: 4px;'></div>
                    <span style='color: #2D1B4E; font-weight: 600;'>🔥 Peak Consumption</span>
                </div>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 24px; height: 24px; background-color: #51CF66; border: 1px solid #2F9E44; border-radius: 4px;'></div>
                    <span style='color: #2D1B4E; font-weight: 600;'>❄️ Lowest Consumption</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.caption(
            f"Key Insights timings in table: Peak at {recent_peak_time.strftime('%Y-%m-%d %H:%M')} (red row) | Low at {recent_low_time.strftime('%Y-%m-%d %H:%M')} (green row)"
        )


st.markdown("---")

# ==================== FOOTER ====================
st.markdown(
    """
    <div class='footer-card'>
        <div class='footer-title'>
            <strong>⚡ PowerPulse Dashboard</strong> | Last Updated: <span id="timestamp"></span>
        </div>
        <div class='footer-subtitle'>
            🌍 Real-time Energy Monitoring | 🤖 AI-Powered Forecasting
        </div>
    </div>
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
    </script>
""",
    unsafe_allow_html=True,
)

if st.button("🔄 Refresh Dashboard", use_container_width=True, key="footer_refresh"):
    st.cache_data.clear()
    st.toast("✅ Dashboard refreshed!")
    st.rerun()
