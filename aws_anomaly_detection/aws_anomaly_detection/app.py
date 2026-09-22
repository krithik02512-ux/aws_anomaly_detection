"""
app.py
========
Main Streamlit dashboard for the AI-Powered AWS Monitoring prototype.

Pipeline implemented on every reading:

    Virtual Sensors -> Data Acquisition -> Preprocessing ->
    ML Anomaly Detection -> Anomaly Score -> Parameter Identification ->
    Alert Engine -> Dashboard

Run with:  streamlit run app.py
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Make sure local packages (ml/, simulation/, utils/) are importable
# regardless of the working directory Streamlit is launched from.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.anomaly_detector import AnomalyDetector, AnomalyResult          # noqa: E402
from simulation.sensor_simulator import SensorSimulator                 # noqa: E402
from utils.data_processing import (                                     # noqa: E402
    compute_trend,
    trend_arrow,
    param_status,
    STATUS_COLORS,
    STATUS_LABELS,
    format_value,
    history_to_display_table,
)

PARAMS = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]
PARAM_LABELS = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind_speed": "Wind Speed",
    "rainfall": "Rainfall",
}
MAX_HISTORY_ROWS = 200
AUTO_REFRESH_SECONDS = 2

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered AWS Monitoring",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        background-color: #ffffff;
    }
    .metric-card h4 {
        margin: 0 0 4px 0;
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .metric-unit {
        font-size: 0.95rem;
        color: #64748b;
        margin-left: 4px;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-top: 6px;
    }
    .system-banner {
        border-radius: 10px;
        padding: 18px 22px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .arch-box {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 16px;
        background-color: #f8fafc;
        text-align: center;
        font-weight: 600;
        color: #0f172a;
        margin: 4px auto;
        width: fit-content;
        min-width: 320px;
    }
    .arch-arrow {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin: -2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources / session state initialisation
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector() -> AnomalyDetector:
    return AnomalyDetector()


def init_state():
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator(seed=7)
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(
            columns=["timestamp"] + PARAMS + ["is_anomaly", "anomaly_score",
                                               "suspicious_param", "explanation",
                                               "is_multi_parameter"]
        )
    if "anomaly_log" not in st.session_state:
        st.session_state.anomaly_log = []
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False
    if "warmed_up" not in st.session_state:
        st.session_state.warmed_up = False


def process_reading(detector: AnomalyDetector, reading: dict) -> AnomalyResult:
    """Runs one reading through the full pipeline and logs it."""
    features = {p: reading[p] for p in PARAMS}
    result = detector.analyze(features)

    row = {
        "timestamp": reading["timestamp"],
        **features,
        "is_anomaly": result.is_anomaly,
        "anomaly_score": result.anomaly_score,
        "suspicious_param": result.suspicious_param,
        "explanation": result.explanation,
        "is_multi_parameter": result.is_multi_parameter,
    }
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([row])], ignore_index=True
    ).tail(MAX_HISTORY_ROWS)

    if result.is_anomaly:
        st.session_state.anomaly_log.append(
            {
                "time": reading["timestamp"],
                "parameter": PARAM_LABELS.get(result.suspicious_param, "Multiple") if not result.is_multi_parameter else "Multiple",
                "value": format_value(result.suspicious_param, reading[result.suspicious_param]) if result.suspicious_param else "-",
                "score": result.anomaly_score,
                "status": "Multi-parameter" if result.is_multi_parameter else "Anomaly",
            }
        )
        st.session_state.anomaly_log = st.session_state.anomaly_log[-100:]

    return result


init_state()
detector = load_detector()
sim: SensorSimulator = st.session_state.simulator

# Warm up with a short run of normal data so the dashboard isn't empty
# the first time it's opened.
if not st.session_state.warmed_up:
    for _ in range(20):
        r = sim.generate_reading()
        process_reading(detector, r)
    st.session_state.warmed_up = True

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛰️ AI-Powered Automatic Weather Station Monitoring")
st.caption("Intelligent Multi-Sensor Anomaly Detection")

tab_dashboard, tab_architecture = st.tabs(["📊 Live Dashboard", "🏗️ System Architecture"])

# ===========================================================================
# SIDEBAR — Simulation Controls
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    st.caption("Virtual AWS sensor simulator — no physical hardware required for this prototype.")

    st.session_state.auto_run = st.toggle(
        "▶ Live auto-refresh", value=st.session_state.auto_run,
        help=f"Automatically generates a new reading every {AUTO_REFRESH_SECONDS}s."
    )

    manual_generate = st.button("🔄 Generate Next Reading", use_container_width=True)

    st.divider()
    st.subheader("Inject Anomaly")
    inject_clicked = None
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌡️ Temperature", use_container_width=True):
            inject_clicked = "temperature"
        if st.button("🧭 Pressure", use_container_width=True):
            inject_clicked = "pressure"
        if st.button("🌧️ Rainfall", use_container_width=True):
            inject_clicked = "rainfall"
    with col_b:
        if st.button("💧 Humidity", use_container_width=True):
            inject_clicked = "humidity"
        if st.button("💨 Wind Speed", use_container_width=True):
            inject_clicked = "wind_speed"

    st.divider()
    reset_clicked = st.button("✅ Reset to Normal", use_container_width=True, type="primary")

    st.divider()
    st.caption(
        "⚠️ Prototype notice: all data shown is simulated/synthetic. "
        "This system does not provide certified meteorological readings "
        "and cannot definitively diagnose a sensor fault vs. a real "
        "weather event."
    )

# ---------------------------------------------------------------------------
# Handle control actions (each triggers exactly one new reading, except
# auto-run which generates one per refresh cycle below)
# ---------------------------------------------------------------------------
reading_triggered = manual_generate

if inject_clicked:
    sim.inject_anomaly(inject_clicked, duration_readings=12)
    reading_triggered = True

if reset_clicked:
    sim.reset()
    reading_triggered = True

if reading_triggered:
    r = sim.generate_reading()
    process_reading(detector, r)

if st.session_state.auto_run:
    r = sim.generate_reading()
    process_reading(detector, r)

history: pd.DataFrame = st.session_state.history
latest = history.iloc[-1] if len(history) else None

# ===========================================================================
# TAB 1 — LIVE DASHBOARD
# ===========================================================================
with tab_dashboard:
    if latest is None:
        st.info("Click **Generate Next Reading** in the sidebar to start the simulation.")
    else:
        # ------------------------------------------------------------
        # System status banner
        # ------------------------------------------------------------
        if latest["is_anomaly"]:
            banner_color = "#fee2e2"
            banner_text_color = "#991b1b"
            label = "🔴 ANOMALY DETECTED"
        else:
            banner_color = "#dcfce7"
            banner_text_color = "#166534"
            label = "🟢 SYSTEM NORMAL"

        st.markdown(
            f"""<div class="system-banner" style="background-color:{banner_color};
            color:{banner_text_color};">{label}
            &nbsp;&nbsp;|&nbsp;&nbsp; Anomaly Score: {latest['anomaly_score']:.1f} / 100</div>""",
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # Sensor cards
        # ------------------------------------------------------------
        cols = st.columns(5)
        for i, param in enumerate(PARAMS):
            value = latest[param]
            z = detector.feature_stats[param]
            z_score = (value - z["mean"]) / (z["std"] if z["std"] > 1e-6 else 1e-6)
            status = param_status(
                param,
                latest["suspicious_param"],
                latest["is_anomaly"],
                z_score,
            )
            trend = compute_trend(history, param)
            unit = SensorSimulator.units()[param]

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{PARAM_LABELS[param]}</h4>
                        <p class="metric-value">{format_value(param, value)}
                            <span class="metric-unit">{unit} {trend_arrow(trend)}</span>
                        </p>
                        <span class="status-badge" style="background-color:{STATUS_COLORS[status]};">
                            {STATUS_LABELS[status]}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

        # ------------------------------------------------------------
        # Live graphs
        # ------------------------------------------------------------
        st.subheader("📈 Live Sensor Trends")
        plot_df = history.tail(60).copy()

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[PARAM_LABELS[p] for p in PARAMS] + [""],
            vertical_spacing=0.12,
        )
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        for param, (row, col) in zip(PARAMS, positions):
            fig.add_trace(
                go.Scatter(
                    x=plot_df["timestamp"], y=plot_df[param],
                    mode="lines", name=PARAM_LABELS[param],
                    line=dict(color="#2563eb", width=2),
                    showlegend=False,
                ),
                row=row, col=col,
            )
            anomalous_pts = plot_df[
                plot_df["is_anomaly"] & (plot_df["suspicious_param"] == param)
            ]
            if len(anomalous_pts):
                fig.add_trace(
                    go.Scatter(
                        x=anomalous_pts["timestamp"], y=anomalous_pts[param],
                        mode="markers", name="Anomaly",
                        marker=dict(color="#dc2626", size=9, symbol="circle"),
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
        fig.update_layout(height=480, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # Anomaly alert panel
        # ------------------------------------------------------------
        if latest["is_anomaly"]:
            st.subheader("🚨 Anomaly Alert")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Detection Time", latest["timestamp"].strftime("%H:%M:%S"))
                param_display = (
                    "Multiple parameters" if latest["is_multi_parameter"]
                    else PARAM_LABELS.get(latest["suspicious_param"], "-")
                )
                c2.metric("Parameter", param_display)
                if latest["suspicious_param"]:
                    c3.metric(
                        "Current Value",
                        f"{format_value(latest['suspicious_param'], latest[latest['suspicious_param']])} "
                        f"{SensorSimulator.units()[latest['suspicious_param']]}",
                    )
                c4.metric("Anomaly Score", f"{latest['anomaly_score']:.1f}")
                status_text = "Requires Verification" if latest["is_multi_parameter"] else "Suspicious Reading"
                st.error(f"**Status:** {status_text}\n\n{latest['explanation']}")

        # ------------------------------------------------------------
        # Anomaly history table
        # ------------------------------------------------------------
        st.subheader("🗂️ Anomaly History")
        table = history_to_display_table(st.session_state.anomaly_log)
        if len(table):
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.caption("No anomalies logged yet. Use the sidebar to inject one.")

# ===========================================================================
# TAB 2 — SYSTEM ARCHITECTURE
# ===========================================================================
with tab_architecture:
    st.subheader("System Architecture")

    steps = [
        "Virtual Sensors",
        "Data Acquisition",
        "Data Preprocessing",
        "ML Anomaly Detection (Isolation Forest)",
        "Anomaly Score (0–100)",
        "Parameter Identification (z-score analysis)",
        "Alert Engine",
        "Dashboard",
    ]
    for i, step in enumerate(steps):
        st.markdown(f'<div class="arch-box">{step}</div>', unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("Why anomaly detection matters for AWS")
    st.markdown(
        """
Automatic Weather Stations run unattended for long periods. A single sensor
can drift, get physically obstructed, or fail outright — and a bad reading
that goes unnoticed can quietly corrupt downstream weather records or
decisions. Checking one parameter against a fixed threshold misses cases
where a sensor is "off" only relative to the *other* sensors at that moment
(e.g. temperature spiking while humidity, pressure, and wind stay put).
Looking at all sensors **together** with a model trained on their normal
joint behaviour makes it possible to catch these cases earlier and point
a technician straight at the sensor most likely responsible.
        """
    )

    st.subheader("How the ML model works (plain-language summary)")
    st.markdown(
        """
1. **Training:** The model is shown thousands of examples of *normal*
   5-sensor readings (temperature, humidity, pressure, wind speed,
   rainfall) and learns what typical combinations look like.
2. **Isolation Forest:** For a new reading, the model tries to "isolate"
   it from the rest of the data using random splits. Normal points take
   many splits to isolate because they sit in a dense region of similar
   points. Unusual points get isolated in very few splits because they
   sit apart from everything else — this "ease of isolation" is turned
   into the anomaly score.
3. **Explainability layer:** If a reading is flagged, the system compares
   each individual sensor to its own learned normal range (a z-score) to
   report *which* sensor is most likely responsible, and flags cases
   where several sensors are off together as needing human verification
   rather than confidently blaming one sensor.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
- This is a **prototype using simulated/synthetic AWS data**, not data
  from certified meteorological hardware.
- It does **not** guarantee accurate diagnosis of a sensor fault versus a
  genuine (if unusual) weather event.
- It is **not** a weather forecasting or disaster-prediction system.
- It is **not** production-deployment ready; it is built to be understood
  and demonstrated within a short competition demo.
        """
    )

if st.session_state.auto_run:
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()
