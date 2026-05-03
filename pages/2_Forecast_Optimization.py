import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Forecast & Optimization", layout="wide")

st.title("🔮 Forecast & Optimization")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F5F3FF 0%, #EDE9FE 100%) !important;
            color: #2D1B4E !important;
            border-right: 1px solid #DDD6FE !important;
        }

        [data-testid="stSidebar"] * {
            color: #2D1B4E !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] a,
        [data-testid="stSidebar"] [data-testid="stPageLink"] p {
            color: #2D1B4E !important;
            font-weight: 700 !important;
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
    </style>
    """,
    unsafe_allow_html=True,
)


def load_df_real():
    # Prefer session state if available, otherwise fall back to live_data.csv
    if "df_real" in st.session_state:
        df = st.session_state.df_real.copy()
    else:
        try:
            df = pd.read_csv(
                "live_data.csv", parse_dates=["Datetime"], index_col="Datetime"
            )
        except Exception:
            st.warning(
                "No session data found and `live_data.csv` is missing. Go to main dashboard or run `live_data_fetch.py`."
            )
            st.stop()

    # Normalize column name to `Energy`
    if "Energy" not in df.columns:
        if "Consumption" in df.columns:
            df = df.rename(columns={"Consumption": "Energy"})
        elif "Forecast" in df.columns:
            df = df.rename(columns={"Forecast": "Energy"})
        else:
            st.error("Data file does not contain `Energy` or `Consumption` column.")
            st.stop()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


df_real = load_df_real()

# ---------------- FORECAST ----------------
# Fixed 24-hour forecast (no slider required)
forecast_hours = 24

model = ARIMA(df_real["Energy"], order=(2, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_hours)

if "Temperature" in df_real.columns:
    temp_series = df_real["Temperature"]
else:
    # Fallback estimate when temperature is unavailable in the input data.
    temp_series = (df_real["Energy"] - 50) / 2

temp_model = ARIMA(temp_series, order=(2, 1, 2))
temp_model_fit = temp_model.fit()
temp_forecast = temp_model_fit.forecast(steps=forecast_hours)

# Keep the displayed temperature forecast in a realistic range for these cities.
temp_forecast = np.clip(temp_forecast, 0, None)

# Start forecasts after the last known timestamp
start_ts = df_real.index[-1] + pd.Timedelta(hours=1)
forecast_df = pd.DataFrame(
    {
        "Datetime": pd.date_range(start=start_ts, periods=forecast_hours, freq="h"),
        "Temperature": temp_forecast,
        "Energy": forecast,
    }
)

# store forecast in session so main app can access it
st.session_state["forecast_df"] = forecast_df

# Plotly forecast graph (historical last 24 + predicted next 24)
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=df_real.tail(24).index,
        y=df_real.tail(24)["Energy"],
        mode="lines+markers",
        name="Historical",
        line=dict(color="#7C3AED", width=3),
        marker=dict(size=6, color="#7C3AED", line=dict(color="#FFFFFF", width=1)),
    )
)

fig2.add_trace(
    go.Scatter(
        x=forecast_df["Datetime"],
        y=forecast_df["Energy"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#A78BFA", width=3, dash="dash"),
        marker=dict(
            size=6,
            symbol="diamond",
            color="#A78BFA",
            line=dict(color="#FFFFFF", width=1),
        ),
        fill="tonexty",
        fillcolor="rgba(167, 139, 250, 0.12)",
    )
)

fig2.update_layout(
    template="plotly",
    height=450,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Time",
    yaxis_title="Predicted Energy (kWh)",
)

st.plotly_chart(fig2, width="stretch")


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

    def highlight_recommendation_rows(row):
        if "Datetime" not in row.index:
            return [""] * len(row)

        try:
            row_time = pd.Timestamp(row["Datetime"]).floor("H")

            if mark_peak_time is not None:
                peak_time_normalized = pd.Timestamp(mark_peak_time).floor("H")
                if row_time == peak_time_normalized:
                    return [
                        "background-color: #FFD1D1 !important; color: #7F1D1D !important; font-weight: 700;"
                    ] * len(row)

            if mark_low_time is not None:
                low_time_normalized = pd.Timestamp(mark_low_time).floor("H")
                if row_time == low_time_normalized:
                    return [
                        "background-color: #D3F9D8 !important; color: #14532D !important; font-weight: 700;"
                    ] * len(row)
        except Exception:
            pass

        return [""] * len(row)

    styled = (
        df_display.style.format(precision=2)
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("width", "100%"),
                        ("background-color", "#FFFFFF"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        (
                            "background",
                            "linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%)",
                        ),
                        ("color", "white"),
                        ("font-weight", "700"),
                        ("border", "1px solid #C4B5FD"),
                        ("padding", "12px"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#F8F6FF"),
                        ("color", "#2D1B4E"),
                        ("border", "1px solid #E9D5FF"),
                        ("padding", "10px"),
                        ("font-weight", "500"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tr:nth-child(even) td",
                    "props": [("background-color", "#FFFFFF")],
                },
                {"selector": "tr:hover td", "props": [("background-color", "#EDE9FE")]},
            ]
        )
        .apply(highlight_recommendation_rows, axis=1)
    )

    st.markdown(
        f"<div class='table-title'>{title}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='table-card'>{styled.to_html()}</div>", unsafe_allow_html=True
    )


# Compute forecast-based key times used for recommendations
forecast_peak_idx = forecast_df["Energy"].idxmax()
forecast_low_idx = forecast_df["Energy"].idxmin()
forecast_peak_time = forecast_df.loc[forecast_peak_idx, "Datetime"]
forecast_low_time = forecast_df.loc[forecast_low_idx, "Datetime"]

# Prepare table (keep only forecast columns)
forecast_table = (
    forecast_df.loc[:, ["Datetime", "Temperature", "Energy"]].head(24).copy()
)

st.caption(
    f"Recommendation times from forecast: Peak at {forecast_peak_time.strftime('%Y-%m-%d %H:%M')} (red row) | Low at {forecast_low_time.strftime('%Y-%m-%d %H:%M')} (green row)"
)

with st.expander("📊 View Detailed Data & Analytics"):
    render_energy_table(
        forecast_table,
        "Energy Forecast (Next 24 hours)",
        highlight_min_color="#FFFFFF",
        mark_peak_time=forecast_peak_time,
        mark_low_time=forecast_low_time,
    )

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

# ---------------- RECOMMENDATIONS ----------------
st.subheader("⚙️ Smart Recommendations")

recommendations = []

if df_real["Energy"].empty:
    st.info("No data available for recommendations.")
else:
    # Use forecast peak/low times for actionable recommendations
    peak_time = forecast_peak_time
    low_time = forecast_low_time

    peak_hour = int(peak_time.hour)
    low_hour = int(low_time.hour)

    # 🔹 Peak-based suggestion
    if peak_hour in range(6, 12):
        recommendations.append(
            f"⚠️ High consumption during morning hours ({peak_hour}:00). Consider shifting heavy usage (washing machine, geyser) to afternoon or night."
        )

    elif peak_hour in range(12, 18):
        recommendations.append(
            f"⚠️ Peak usage in afternoon ({peak_hour}:00). This may be due to cooling appliances. Try optimizing AC usage or using energy-efficient modes."
        )

    else:
        recommendations.append(
            f"⚠️ Peak usage during evening/night ({peak_hour}:00). Avoid running multiple heavy appliances at the same time."
        )

    # 🔹 Low usage suggestion
    if low_hour in range(0, 6):
        recommendations.append(
            f"💡 Energy usage is lowest at {low_hour}:00. This is the best time to schedule high-power tasks to save cost."
        )
    else:
        recommendations.append(
            f"💡 Lowest consumption occurs at {low_hour}:00. Consider utilizing this period for energy-intensive activities."
        )

    # 🔹 Stability suggestion
    std_dev = df_real["Energy"].std()

    if std_dev < 3:
        recommendations.append(
            "📊 Your energy usage is stable. No major fluctuations detected."
        )
    else:
        recommendations.append(
            "📊 Energy usage shows fluctuations. Try distributing usage more evenly across the day."
        )

    # 🔹 Forecast trend suggestion
    if forecast_df["Energy"].mean() > df_real["Energy"].mean():
        recommendations.append(
            "📈 Forecast indicates increasing energy usage. Plan ahead to avoid peak load costs."
        )
    else:
        recommendations.append(
            "📉 Energy usage is expected to remain stable or decrease."
        )

    # 🔹 Display all
    for rec in recommendations:
        st.info(rec)
