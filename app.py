
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "daily_energy_features.csv"
MODEL_FILE = BASE_DIR / "models" / "best_model.joblib"
METRICS_FILE = BASE_DIR / "outputs" / "model_evaluation.csv"
PREDICTIONS_FILE = BASE_DIR / "outputs" / "test_predictions.csv"
FORECAST_FILE = BASE_DIR / "outputs" / "future_forecast.csv"

BG = "#071225"
PANEL = "#101E38"
PANEL_LIGHT = "#152544"
BORDER = "#263A5D"
TEXT = "#F2F5FF"
MUTED = "#A8B7D5"
PURPLE = "#7657FF"
CYAN = "#25B9F4"
GREEN = "#A8E66B"
PINK = "#FF557A"
ORANGE = "#FF8C70"

st.set_page_config(
    page_title="EnergyAI | Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: dark;
    }}
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{
        background: rgba(7,18,37,0.95);
    }}
    [data-testid="stSidebar"] {{
        background: #0B1830;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}
    .block-container {{
        max-width: 1600px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, p, label {{
        color: {TEXT};
    }}
    h1 {{
        font-size: 1.8rem;
        font-weight: 750;
    }}
    h2 {{
        font-size: 1.35rem;
        font-weight: 700;
    }}
    h3 {{
        font-size: 1.05rem;
        font-weight: 650;
    }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {PANEL_LIGHT}, {PANEL});
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 16px 18px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT};
    }}
    div[data-testid="stPlotlyChart"] {{
        background: transparent;
        border-radius: 12px;
    }}
    div[data-testid="stTabs"] button {{
        color: {MUTED};
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {TEXT};
        border-bottom-color: {PURPLE};
    }}
    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"],
    div[data-testid="stDateInput"],
    div[data-testid="stSlider"],
    div[data-testid="stRadio"] {{
        color: {TEXT};
    }}
    .hero {{
        background: linear-gradient(115deg, #12264A, #101A34 65%, #17264A);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }}
    .eyebrow {{
        color: {CYAN};
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 7px;
    }}
    .hero-title {{
        font-size: 27px;
        font-weight: 800;
        color: {TEXT};
        margin-bottom: 7px;
    }}
    .muted {{
        color: {MUTED};
        font-size: 13px;
    }}
    .section-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 17px;
        min-height: 100px;
    }}
    .section-label {{
        font-size: 12px;
        color: {MUTED};
        margin-bottom: 8px;
    }}
    .section-value {{
        font-size: 25px;
        font-weight: 750;
        color: {TEXT};
    }}
    .status-pill {{
        display: inline-block;
        background: #183A35;
        color: {GREEN};
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }}
    div.stButton > button,
    div.stDownloadButton > button {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        background: linear-gradient(100deg, #6344F5, #4936DD);
        color: white;
        font-weight: 650;
    }}
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {{
        border-color: {CYAN};
        color: white;
    }}
    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("The daily dataset is empty.")

    data["daily_consumption_kwh"] = pd.to_numeric(
        data["daily_consumption_kwh"], errors="coerce"
    )
    data = data.dropna(
        subset=["date", "daily_consumption_kwh"]
    ).reset_index(drop=True)

    if "day_of_week" not in data.columns:
        data["day_of_week"] = data["date"].dt.dayofweek

    return data


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def style_chart(fig, height=320, legend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Arial", size=11),
        title_font=dict(color=TEXT, size=15),
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=MUTED, size=10),
        ) if legend else dict(visible=False),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#233451",
        zeroline=False,
        linecolor=BORDER,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#233451",
        zeroline=False,
        linecolor=BORDER,
    )
    return fig


def money(value):
    return f"₹{value:,.2f}"


def card_heading(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def metric_or_na(frame, column, model_name=None):
    if frame.empty or column not in frame.columns:
        return None

    selected = frame

    if model_name and "model" in frame.columns:
        selected = frame[
            frame["model"].astype(str).str.lower()
            == model_name.lower()
        ]

    if selected.empty:
        return None

    try:
        return float(selected.iloc[0][column])
    except (ValueError, TypeError):
        return None


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the daily dataset: {error}")
    st.stop()

model = load_model()
metrics = load_csv(METRICS_FILE)
test_predictions = load_csv(PREDICTIONS_FILE)
forecast = load_csv(FORECAST_FILE)

if not forecast.empty:
    if "date" in forecast.columns:
        forecast["date"] = pd.to_datetime(
            forecast["date"], errors="coerce"
        )
    if "predicted_consumption_kwh" in forecast.columns:
        forecast["predicted_consumption_kwh"] = pd.to_numeric(
            forecast["predicted_consumption_kwh"], errors="coerce"
        )
    forecast = forecast.dropna(
        subset=[
            column for column in [
                "date",
                "predicted_consumption_kwh",
            ]
            if column in forecast.columns
        ]
    )

if not metrics.empty:
    for column in ["MAE_kWh", "RMSE_kWh"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(
                metrics[column], errors="coerce"
            )

if not test_predictions.empty:
    for column in test_predictions.columns:
        if column.lower() in [
            "actual",
            "actual_kwh",
            "actual_consumption_kwh",
            "predicted",
            "predicted_kwh",
            "predicted_consumption_kwh",
            "daily_consumption_kwh",
        ]:
            test_predictions[column] = pd.to_numeric(
                test_predictions[column], errors="coerce"
            )

st.sidebar.markdown(
    f"""
    <div style="padding:8px 0 22px 0;">
        <div style="font-size:25px;font-weight:850;">
            <span style="color:{CYAN};">⚡</span> EnergyAI
        </div>
        <div style="font-size:11px;color:{MUTED};">
            CONSUMPTION ANALYTICS
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Historical Analysis",
        "Future Forecast",
        "Model Evaluation",
        "Energy Budget",
        "Tariff & Recommendations",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**System status**")
if model is not None:
    st.sidebar.success("Forecasting model loaded")
else:
    st.sidebar.warning("Saved model not found")

st.sidebar.caption(f"Dataset records: {len(df):,}")
st.sidebar.caption(
    f"Last available date: {df['date'].max().strftime('%d %b %Y')}"
)
st.sidebar.divider()
st.sidebar.caption(
    "Academic ML project · Forecasts are estimates."
)

latest_date = df["date"].max()
latest_usage = float(df.iloc[-1]["daily_consumption_kwh"])
average_usage = float(df["daily_consumption_kwh"].mean())
total_usage = float(df["daily_consumption_kwh"].sum())

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">SMART ENERGY ANALYTICS</div>
        <div class="hero-title">Energy Consumption Dashboard</div>
        <div class="muted">
            Explore consumption trends, forecast energy usage, and
            estimate costs using your own electricity tariff.
            &nbsp; | &nbsp; Data through {latest_date:%d %b %Y}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Dashboard":
    start_default = max(
        df["date"].min().date(),
        (latest_date - pd.Timedelta(days=6)).date(),
    )

    with st.container(border=True):
        filter_col, view_col = st.columns([2, 1])
        with filter_col:
            dashboard_range = st.date_input(
                "Dashboard date range",
                value=(start_default, latest_date.date()),
                min_value=df["date"].min().date(),
                max_value=latest_date.date(),
                key="dashboard_range",
            )
        with view_col:
            chart_mode = st.selectbox(
                "Cost chart grouping",
                ["Daily", "Weekly", "Monthly"],
                index=0,
            )

    if isinstance(dashboard_range, (tuple, list)) and len(dashboard_range) == 2:
        start_date, end_date = dashboard_range
        view_df = df[
            (df["date"].dt.date >= start_date)
            & (df["date"].dt.date <= end_date)
        ].copy()
    else:
        view_df = df.copy()

    if view_df.empty:
        st.warning("No readings are available for this date range.")
        st.stop()

    daily_average = float(view_df["daily_consumption_kwh"].mean())
    period_usage = float(view_df["daily_consumption_kwh"].sum())
    tariff = st.session_state.get("dashboard_tariff", 8.0)
    budget = st.session_state.get("dashboard_budget", 2500.0)
    estimated_cost = period_usage * tariff

    top_left, top_right = st.columns([1, 2.8], gap="medium")

    with top_left:
        card_heading("Cost overview", "Estimated cost based on your tariff")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=estimated_cost,
                number={
                    "prefix": "₹",
                    "valueformat": ",.2f",
                    "font": {"size": 29, "color": TEXT},
                },
                title={
                    "text": "Estimated period cost",
                    "font": {"size": 12, "color": MUTED},
                },
                gauge={
                    "axis": {
                        "range": [0, max(budget, estimated_cost, 1)],
                        "tickcolor": MUTED,
                        "tickfont": {"color": MUTED, "size": 9},
                    },
                    "bar": {"color": PURPLE, "thickness": 0.26},
                    "bgcolor": PANEL,
                    "borderwidth": 0,
                    "steps": [
                        {
                            "range": [
                                0,
                                max(budget, estimated_cost, 1),
                            ],
                            "color": PANEL_LIGHT,
                        }
                    ],
                    "threshold": {
                        "line": {"color": GREEN, "width": 3},
                        "thickness": 0.8,
                        "value": min(
                            budget, max(budget, estimated_cost, 1)
                        ),
                    },
                },
            )
        )
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT},
            height=240,
            margin=dict(l=15, r=15, t=40, b=5),
        )
        st.plotly_chart(
            gauge, use_container_width=True, key="dashboard_gauge"
        )

        budget_col, usage_col = st.columns(2)
        budget_col.metric("Budget", money(budget))
        usage_col.metric("Usage", f"{period_usage:,.1f} kWh")
        st.caption(
            f"Average daily consumption: {daily_average:.2f} kWh"
        )

    with top_right:
        card_heading(
            "Weekly Cost Overview",
            "Stacked bars show estimated cost by time period.",
        )

        cost_df = view_df[["date", "daily_consumption_kwh"]].copy()
        cost_df["Electricity"] = (
            cost_df["daily_consumption_kwh"] * tariff
        )

        if chart_mode == "Weekly":
            cost_df["Period"] = (
                cost_df["date"].dt.to_period("W").astype(str)
            )
        elif chart_mode == "Monthly":
            cost_df["Period"] = (
                cost_df["date"].dt.to_period("M").astype(str)
            )
        else:
            cost_df["Period"] = cost_df["date"].dt.strftime("%d %b")

        grouped_cost = (
            cost_df.groupby("Period", as_index=False)["Electricity"]
            .sum()
        )
        grouped_cost["Other charges (example)"] = (
            grouped_cost["Electricity"] * 0.05
        )

        cost_fig = go.Figure()
        cost_fig.add_trace(
            go.Bar(
                x=grouped_cost["Period"],
                y=grouped_cost["Electricity"],
                name="Electricity estimate",
                marker_color=PURPLE,
                hovertemplate="%{x}<br>Electricity: ₹%{y:.2f}<extra></extra>",
            )
        )
        cost_fig.add_trace(
            go.Bar(
                x=grouped_cost["Period"],
                y=grouped_cost["Other charges (example)"],
                name="Illustrative extra 5%",
                marker_color=GREEN,
                hovertemplate="%{x}<br>Illustrative extra: ₹%{y:.2f}<extra></extra>",
            )
        )
        cost_fig.update_layout(
            barmode="stack",
            xaxis_title=None,
            yaxis_title="Estimated cost (₹)",
        )
        style_chart(cost_fig, height=350)
        st.plotly_chart(
            cost_fig, use_container_width=True, key="cost_overview"
        )
        st.caption(
            "Extra charges are illustrative only and are not actual utility fees."
        )

    st.divider()

    consumption_left, consumption_right = st.columns([1, 2.8], gap="medium")

    with consumption_left:
        card_heading("Energy Consumption", "Share of usage across selected dates")

        donut = go.Figure(
            go.Pie(
                labels=["Selected period consumption"],
                values=[period_usage],
                hole=0.68,
                marker_colors=[CYAN],
                textinfo="percent",
                hovertemplate=(
                    "Selected period<br>%{value:.2f} kWh<extra></extra>"
                ),
            )
        )
        donut.add_annotation(
            text=f"Total<br>{period_usage:.1f} kWh",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT, size=15),
        )
        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            showlegend=False,
            height=285,
            margin=dict(l=5, r=5, t=10, b=5),
        )
        st.plotly_chart(
            donut, use_container_width=True, key="consumption_donut"
        )
        st.caption(
            "A single consumption category is shown because appliance-level "
            "measurements are not available in the current dataset."
        )

    with consumption_right:
        card_heading(
            "Consumption Trend",
            "Daily electricity use in the selected period",
        )
        trend = go.Figure()
        trend.add_trace(
            go.Scatter(
                x=view_df["date"],
                y=view_df["daily_consumption_kwh"],
                mode="lines+markers",
                name="Actual consumption",
                line=dict(color=GREEN, width=2.5, shape="spline"),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(168,230,107,0.10)",
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} kWh<extra></extra>",
            )
        )

        if "rolling_mean_7_days" in view_df.columns:
            trend.add_trace(
                go.Scatter(
                    x=view_df["date"],
                    y=view_df["rolling_mean_7_days"],
                    mode="lines",
                    name="7-day rolling average",
                    line=dict(color=CYAN, width=2, dash="dot"),
                )
            )

        trend.update_layout(
            xaxis_title=None,
            yaxis_title="Consumption (kWh)",
        )
        style_chart(trend, height=310)
        st.plotly_chart(
            trend, use_container_width=True, key="energy_trend"
        )

    st.divider()

    lower_left, lower_middle, lower_right = st.columns(
        [1, 1.15, 1.8], gap="medium"
    )

    with lower_left:
        card_heading("Electricity")
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Entered tariff</div>
                <div class="section-value">{money(tariff)} <span style="font-size:12px">/kWh</span></div>
                <div class="muted">Estimated usage cost</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.number_input(
            "Tariff per kWh (₹)",
            min_value=0.0,
            max_value=1000.0,
            value=float(tariff),
            step=0.5,
            key="dashboard_tariff",
        )

    with lower_middle:
        card_heading("Current Tariff")
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">Tariff source</div>
                <div class="section-value" style="font-size:19px">User supplied</div>
                <hr style="border-color:{BORDER}">
                <div class="muted">Electricity rate</div>
                <div style="font-size:18px;font-weight:700">{money(tariff)} / kWh</div>
                <div class="muted" style="margin-top:10px">
                    No live supplier tariff is connected.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lower_right:
        card_heading("Recommendations", "Based on your recorded consumption")
        if daily_average > average_usage * 1.1:
            st.warning(
                "Consumption in the selected period is above the full-dataset "
                "average. Review high-usage days."
            )
        elif daily_average < average_usage * 0.9:
            st.success(
                "Consumption in the selected period is below the "
                "full-dataset average."
            )
        else:
            st.info(
                "Consumption is close to the full-dataset average. "
                "Try comparing different date ranges."
            )

        if estimated_cost > budget:
            st.error(
                f"Estimated cost is {money(estimated_cost - budget)} "
                "above your budget."
            )
        else:
            st.success(
                f"Estimated budget remaining: {money(budget - estimated_cost)}."
            )

        st.caption(
            "These are data-based prompts, not measured appliance diagnostics "
            "or guaranteed savings."
        )

    st.divider()
    card_heading("Latest Recorded Readings")
    st.dataframe(
        view_df.tail(10).sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Selected Historical Data",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="selected_energy_history.csv",
        mime="text/csv",
        key="dashboard_download_history",
    )


elif page == "Historical Analysis":
    card_heading(
        "Historical Energy Analysis",
        "Filter your records and compare consumption patterns.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start = st.date_input(
            "Start date",
            value=df["date"].min().date(),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="history_start",
        )
    with col_b:
        end = st.date_input(
            "End date",
            value=df["date"].max().date(),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="history_end",
        )

    if start > end:
        st.error("Start date must be on or before the end date.")
        st.stop()

    filtered = df[
        (df["date"].dt.date >= start)
        & (df["date"].dt.date <= end)
    ].copy()

    weekdays = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    filtered["weekday"] = filtered["date"].dt.dayofweek.map(weekdays)

    selected_days = st.multiselect(
        "Filter weekdays",
        list(weekdays.values()),
        default=list(weekdays.values()),
    )
    filtered = filtered[filtered["weekday"].isin(selected_days)]

    if filtered.empty:
        st.warning("No readings match the selected filters.")
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("Total consumption", f"{filtered['daily_consumption_kwh'].sum():,.2f} kWh")
    b.metric("Daily average", f"{filtered['daily_consumption_kwh'].mean():,.2f} kWh")
    c.metric("Maximum daily use", f"{filtered['daily_consumption_kwh'].max():,.2f} kWh")
    c_delta = float(filtered["daily_consumption_kwh"].max() - filtered["daily_consumption_kwh"].min())
    d.metric("Records", f"{len(filtered):,}")

    fig = px.area(
        filtered,
        x="date",
        y="daily_consumption_kwh",
        title="Consumption Over Selected Period",
        labels={
            "date": "Date",
            "daily_consumption_kwh": "Consumption (kWh)",
        },
    )
    fig.update_traces(
        line_color=CYAN,
        fillcolor="rgba(37,185,244,0.15)",
    )
    style_chart(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    weekday_data = (
        filtered.groupby("weekday", as_index=False)
        ["daily_consumption_kwh"].mean()
    )
    weekday_data["weekday"] = pd.Categorical(
        weekday_data["weekday"],
        categories=list(weekdays.values()),
        ordered=True,
    )
    weekday_data = weekday_data.sort_values("weekday")

    weekday_fig = px.bar(
        weekday_data,
        x="weekday",
        y="daily_consumption_kwh",
        title="Average Consumption by Day of Week",
        labels={
            "weekday": "Day",
            "daily_consumption_kwh": "Average consumption (kWh)",
        },
        color_discrete_sequence=[PURPLE],
    )
    style_chart(weekday_fig, height=330, legend=False)
    st.plotly_chart(weekday_fig, use_container_width=True)

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Filtered History",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_energy_history.csv",
        mime="text/csv",
    )


elif page == "Future Forecast":
    card_heading(
        "Future Electricity Forecast",
        "Predictions from your saved forecast file.",
    )

    if forecast.empty:
        st.warning(
            "No valid forecast found. Run "
            "python src/06_forecast_future.py to generate it."
        )
        st.stop()

    if not {
        "date",
        "predicted_consumption_kwh",
    }.issubset(forecast.columns):
        st.error(
            "The forecast CSV must contain date and "
            "predicted_consumption_kwh columns."
        )
        st.stop()

    horizon = st.select_slider(
        "Forecast horizon",
        options=list(range(1, len(forecast) + 1)),
        value=min(7, len(forecast)),
    )
    selected_forecast = forecast.head(horizon).copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast days", len(selected_forecast))
    col2.metric(
        "Forecast total",
        f"{selected_forecast['predicted_consumption_kwh'].sum():,.2f} kWh",
    )
    col3.metric(
        "Daily forecast average",
        f"{selected_forecast['predicted_consumption_kwh'].mean():,.2f} kWh",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["date"],
            y=selected_forecast["predicted_consumption_kwh"],
            mode="lines+markers",
            name="Predicted consumption",
            line=dict(color=PURPLE, width=3),
            marker=dict(size=8, color=CYAN),
            fill="tozeroy",
            fillcolor="rgba(118,87,255,0.12)",
        )
    )

    fig.add_hline(
        y=average_usage,
        line_dash="dash",
        line_color=GREEN,
        annotation_text="Historical daily average",
    )
    fig.update_layout(
        title="Predicted Daily Electricity Consumption",
        xaxis_title="Date",
        yaxis_title="Predicted consumption (kWh)",
    )
    style_chart(fig, height=390)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        selected_forecast,
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Forecast CSV",
        data=selected_forecast.to_csv(index=False).encode("utf-8"),
        file_name="future_forecast.csv",
        mime="text/csv",
    )


elif page == "Model Evaluation":
    card_heading(
        "Machine Learning Model Evaluation",
        "Review the saved model comparison and test predictions.",
    )

    if metrics.empty:
        st.warning(
            "No model evaluation file found. Run "
            "python src/04_train_models.py first."
        )
    else:
        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        mae_col, rmse_col = st.columns(2)

        if {"model", "MAE_kWh"}.issubset(metrics.columns):
            plot_mae = metrics.dropna(subset=["MAE_kWh"])
            if not plot_mae.empty:
                with mae_col:
                    fig = px.bar(
                        plot_mae,
                        x="model",
                        y="MAE_kWh",
                        title="Mean Absolute Error",
                        color_discrete_sequence=[CYAN],
                        labels={
                            "model": "Model",
                            "MAE_kWh": "MAE (kWh)",
                        },
                    )
                    style_chart(fig, height=330, legend=False)
                    st.plotly_chart(fig, use_container_width=True)

        if {"model", "RMSE_kWh"}.issubset(metrics.columns):
            plot_rmse = metrics.dropna(subset=["RMSE_kWh"])
            if not plot_rmse.empty:
                with rmse_col:
                    fig = px.bar(
                        plot_rmse,
                        x="model",
                        y="RMSE_kWh",
                        title="Root Mean Squared Error",
                        color_discrete_sequence=[PURPLE],
                        labels={
                            "model": "Model",
                            "RMSE_kWh": "RMSE (kWh)",
                        },
                    )
                    style_chart(fig, height=330, legend=False)
                    st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Lower MAE and RMSE values generally indicate smaller prediction "
            "errors on the evaluated test period."
        )

    if not test_predictions.empty:
        st.divider()
        card_heading(
            "Test-Period Predictions",
            "Inspect saved predictions. Select columns based on your CSV schema.",
        )
        st.dataframe(
            test_predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Test Predictions",
            data=test_predictions.to_csv(index=False).encode("utf-8"),
            file_name="test_predictions.csv",
            mime="text/csv",
        )


elif page == "Energy Budget":
    card_heading(
        "Energy Budget Simulator",
        "Change the assumptions to estimate energy costs for a selected period.",
    )

    with st.container(border=True):
        source = st.radio(
            "Consumption source",
            ["Historical average", "Forecast average"],
            horizontal=True,
        )

        if source == "Forecast average" and not forecast.empty and "predicted_consumption_kwh" in forecast.columns:
            values = pd.to_numeric(
                forecast["predicted_consumption_kwh"], errors="coerce"
            ).dropna()
            if values.empty:
                daily_kwh = average_usage
                source_label = "Historical average (forecast unavailable)"
            else:
                daily_kwh = float(values.mean())
                source_label = "Forecast average"
        else:
            daily_kwh = average_usage
            source_label = "Historical average"

        col1, col2 = st.columns(2)

        with col1:
            tariff = st.slider(
                "Electricity tariff (₹ per kWh)",
                min_value=1.0,
                max_value=30.0,
                value=8.0,
                step=0.5,
            )
            days = st.slider(
                "Estimate period (days)",
                min_value=1,
                max_value=90,
                value=30,
            )

        with col2:
            budget = st.slider(
                "Energy budget (₹)",
                min_value=500,
                max_value=20000,
                value=2500,
                step=250,
            )
            usage_adjustment = st.slider(
                "Consumption adjustment (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                help="Simulate lower or higher usage without changing your model.",
            )

        adjusted_daily_kwh = daily_kwh * (1 + usage_adjustment / 100)
        estimated_units = adjusted_daily_kwh * days
        estimated_cost = estimated_units * tariff
        budget_remaining = budget - estimated_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily consumption", f"{adjusted_daily_kwh:.2f} kWh")
    col2.metric("Estimated units", f"{estimated_units:.2f} kWh")
    col3.metric("Projected cost", money(estimated_cost))
    col4.metric("Budget", money(budget))

    st.caption(f"Calculation source: {source_label}")

    progress = min(estimated_cost / budget, 1.0) if budget > 0 else 1.0
    st.progress(progress, text=f"{estimated_cost / budget * 100:.1f}% of budget used")

    if budget_remaining < 0:
        st.error(
            f"Projected cost exceeds your budget by {money(abs(budget_remaining))}."
        )
    else:
        st.success(
            f"Projected cost is within budget. Remaining: {money(budget_remaining)}."
        )

    comparison = pd.DataFrame(
        {
            "Category": ["Budget", "Projected cost"],
            "Amount": [budget, estimated_cost],
        }
    )
    fig = px.bar(
        comparison,
        x="Category",
        y="Amount",
        color="Category",
        color_discrete_map={
            "Budget": CYAN,
            "Projected cost": PINK if estimated_cost > budget else PURPLE,
        },
        title="Budget vs Projected Cost",
    )
    style_chart(fig, height=330, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    daily_cost = pd.DataFrame(
        {
            "Day": np.arange(1, days + 1),
            "Estimated cumulative cost": (
                np.arange(1, days + 1) * adjusted_daily_kwh * tariff
            ),
        }
    )
    fig = px.area(
        daily_cost,
        x="Day",
        y="Estimated cumulative cost",
        title="Cumulative Estimated Cost",
    )
    fig.update_traces(
        line_color=GREEN,
        fillcolor="rgba(168,230,107,0.12)",
    )
    style_chart(fig, height=310, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This is a scenario estimate, not an actual bill. Taxes, fixed charges, "
        "tariff slabs, and supplier-specific fees are not included."
    )


elif page == "Tariff & Recommendations":
    card_heading(
        "Tariff & Energy Recommendations",
        "Explore estimated tariff costs and consumption-based suggestions.",
    )

    st.info(
        "The current dataset records total daily consumption. It does not "
        "provide verified live supplier tariffs, gas usage, appliance-level "
        "measurements, or supplier switching offers."
    )

    col1, col2 = st.columns(2)

    with col1:
        card_heading("Electricity Tariff Calculator")
        rate = st.number_input(
            "Your electricity rate (₹/kWh)",
            min_value=0.0,
            max_value=1000.0,
            value=8.0,
            step=0.5,
            key="recommendation_tariff",
        )
        monthly_days = st.slider(
            "Billing period",
            min_value=1,
            max_value=31,
            value=30,
            key="tariff_days",
        )
        monthly_usage = average_usage * monthly_days
        st.metric("Estimated units", f"{monthly_usage:,.2f} kWh")
        st.metric("Estimated bill", money(monthly_usage * rate))

    with col2:
        card_heading("Consumption Insights")
        if average_usage > 0:
            st.metric("Historical daily average", f"{average_usage:.2f} kWh")
            st.metric("Historical total", f"{total_usage:,.2f} kWh")

            high_days = df[
                df["daily_consumption_kwh"] > average_usage * 1.25
            ].copy()

            if not high_days.empty:
                st.warning(
                    f"{len(high_days)} recorded days were more than 25% above "
                    "the overall daily average. Review these dates for possible "
                    "usage patterns."
                )
                st.dataframe(
                    high_days[["date", "daily_consumption_kwh"]].tail(10),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No recorded day exceeded 125% of the historical average."
                )

            st.markdown("**Practical energy-saving checks**")
            st.markdown(
                "- Compare consumption across similar weekdays.\n"
                "- Investigate unusually high-usage days.\n"
                "- Review your actual electricity tariff and bill.\n"
                "- Revisit the budget simulator to test lower-usage scenarios."
            )

    st.divider()
    card_heading("Electricity Price Scenario")
    price_range = st.slider(
        "Compare tariff assumptions (₹/kWh)",
        min_value=1.0,
        max_value=30.0,
        value=(5.0, 12.0),
        step=0.5,
    )
    rates = np.arange(
        price_range[0],
        price_range[1] + 0.001,
        0.5,
    )
    rate_scenario = pd.DataFrame(
        {
            "Tariff (₹/kWh)": rates,
            "Estimated period cost (₹)": rates * average_usage * 30,
        }
    )
    fig = px.line(
        rate_scenario,
        x="Tariff (₹/kWh)",
        y="Estimated period cost (₹)",
        markers=True,
        title="How tariff assumptions affect a 30-day estimate",
    )
    fig.update_traces(line_color=CYAN)
    style_chart(fig, height=350, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This comparison is a tariff scenario, not a live supplier ranking. "
        "Gas and appliance-level charts require corresponding source data."
    )


st.divider()
st.caption(
    "EnergyAI · Energy Consumption Forecasting · "
    "Historical measurements and model forecasts are distinct. "
    "Cost calculations depend on user-entered tariff assumptions."
)
