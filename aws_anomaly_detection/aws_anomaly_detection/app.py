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
from utils.alert_engine import (                                        # noqa: E402
    build_email_notification,
    build_sms_notification,
    severity_label,
)
from utils.weather_check import get_real_weather, classify_anomaly_source  # noqa: E402

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

# Virtual AWS network: each station gets its own simulator/history/logs so
# they operate completely independently, the way real field stations would.
# The seed differs per station only so their live demo data streams look
# distinct from each other (not to change what counts as "normal" -
# they all share the same trained model and learned normal ranges).
STATIONS = [
    {"id": "AWS-STN-01", "name": "AWS-STN-01 — Chennai", "seed": 7},
    {"id": "AWS-STN-02", "name": "AWS-STN-02 — Coimbatore", "seed": 21},
    {"id": "AWS-STN-03", "name": "AWS-STN-03 — Madurai", "seed": 35},
]
STATION_IDS = [s["id"] for s in STATIONS]
STATION_LABELS = {s["id"]: s["name"] for s in STATIONS}

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


@st.cache_data(ttl=600, show_spinner=False)
def fetch_real_weather_cached(station_id: str):
    """Cached for 10 minutes per station so the live 2-second dashboard
    refresh doesn't hammer the OpenWeatherMap API or its free-tier rate
    limit. Real weather doesn't change meaningfully within 10 minutes
    anyway."""
    try:
        api_key = st.secrets.get("OPENWEATHER_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    return get_real_weather(station_id, api_key)


def init_state():
    if "stations" not in st.session_state:
        st.session_state.stations = {}
    if "selected_station" not in st.session_state:
        st.session_state.selected_station = STATION_IDS[0]


def get_station_state(station_id: str) -> dict:
    """Lazily creates and returns the isolated state (simulator, history,
    anomaly log, notification log) for one AWS station."""
    if station_id not in st.session_state.stations:
        seed = next(s["seed"] for s in STATIONS if s["id"] == station_id)
        st.session_state.stations[station_id] = {
            "simulator": SensorSimulator(seed=seed),
            "history": pd.DataFrame(
                columns=["timestamp"] + PARAMS + ["is_anomaly", "anomaly_score",
                                                   "suspicious_param", "explanation",
                                                   "is_multi_parameter"]
            ),
            "anomaly_log": [],
            "notification_log": [],
            "auto_run": False,
            "warmed_up": False,
        }
    return st.session_state.stations[station_id]


def process_reading(detector: AnomalyDetector, station: dict, reading: dict) -> AnomalyResult:
    """Runs one reading through the full pipeline and logs it against the
    given station's own history/anomaly/notification logs."""
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
    station["history"] = pd.concat(
        [station["history"], pd.DataFrame([row])], ignore_index=True
    ).tail(MAX_HISTORY_ROWS)

    if result.is_anomaly:
        station["anomaly_log"].append(
            {
                "time": reading["timestamp"],
                "parameter": PARAM_LABELS.get(result.suspicious_param, "Multiple") if not result.is_multi_parameter else "Multiple",
                "value": format_value(result.suspicious_param, reading[result.suspicious_param]) if result.suspicious_param else "-",
                "score": result.anomaly_score,
                "status": "Multi-parameter" if result.is_multi_parameter else "Anomaly",
            }
        )
        station["anomaly_log"] = station["anomaly_log"][-100:]

        # --- Alert Engine: build the notification that WOULD be sent ---
        param_label = PARAM_LABELS.get(result.suspicious_param, "Unknown") if result.suspicious_param else "Unknown"
        value_display = (
            f"{format_value(result.suspicious_param, reading[result.suspicious_param])} "
            f"{SensorSimulator.units()[result.suspicious_param]}"
        ) if result.suspicious_param else "-"

        email = build_email_notification(
            parameter=result.suspicious_param,
            parameter_label=param_label,
            value_display=value_display,
            anomaly_score=result.anomaly_score,
            explanation=result.explanation,
            detected_at=reading["timestamp"],
            is_multi_parameter=result.is_multi_parameter,
        )
        sms = build_sms_notification(
            parameter_label=param_label,
            anomaly_score=result.anomaly_score,
            is_multi_parameter=result.is_multi_parameter,
        )
        station["notification_log"].append(
            {"time": reading["timestamp"], "email": email, "sms": sms}
        )
        station["notification_log"] = station["notification_log"][-50:]

    return result


init_state()
detector = load_detector()
station = get_station_state(st.session_state.selected_station)
sim: SensorSimulator = station["simulator"]

# Warm up with a short run of normal data so the dashboard isn't empty
# the first time this station is opened.
if not station["warmed_up"]:
    for _ in range(20):
        r = sim.generate_reading()
        process_reading(detector, station, r)
    station["warmed_up"] = True

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛰️ AI-Powered Automatic Weather Station Monitoring")
st.caption(
    f"Intelligent Multi-Sensor Anomaly Detection  |  "
    f"📡 Monitoring: **{STATION_LABELS[st.session_state.selected_station]}**"
)

tab_dashboard, tab_architecture, tab_metrics = st.tabs(
    ["📊 Live Dashboard", "🏗️ System Architecture", "📐 Model Performance"]
)

# ===========================================================================
# SIDEBAR — Simulation Controls
# ===========================================================================
with st.sidebar:
    st.header("🛰️ AWS Station Network")
    selected = st.selectbox(
        "Select station", options=STATION_IDS,
        format_func=lambda sid: STATION_LABELS[sid],
        index=STATION_IDS.index(st.session_state.selected_station),
        help="Each station runs its own independent virtual sensors and "
             "anomaly history, sharing the same trained ML model."
    )
    if selected != st.session_state.selected_station:
        st.session_state.selected_station = selected
        st.rerun()

    st.divider()
    st.header("⚙️ Simulation Controls")
    st.caption("Virtual AWS sensor simulator — no physical hardware required for this prototype.")

    station["auto_run"] = st.toggle(
        "▶ Live auto-refresh", value=station["auto_run"],
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
# Handle manual control actions (buttons already cause a normal full rerun
# when clicked, which is expected/fine — this is NOT the auto-refresh path).
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
    process_reading(detector, station, r)

# ===========================================================================
# TAB 1 — LIVE DASHBOARD (auto-refreshing fragment)
# ---------------------------------------------------------------------------
# FIX for the "blinking" bug: this used to be driven by a bottom-of-script
# `time.sleep(AUTO_REFRESH_SECONDS); st.rerun()` block, which reran the
# ENTIRE app (sidebar, header, tabs, everything) every 2 seconds — causing
# the whole page to visibly flash/blink on every refresh.
#
# `st.fragment(run_every=...)` reruns ONLY this function on its own timer,
# leaving the rest of the page untouched, so only the live numbers/graph
# update quietly instead of the whole screen blinking.
# ===========================================================================
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def live_dashboard_fragment():
    station = get_station_state(st.session_state.selected_station)

    if station["auto_run"]:
        r = station["simulator"].generate_reading()
        process_reading(detector, station, r)

    history: pd.DataFrame = station["history"]
    latest = history.iloc[-1] if len(history) else None

    with tab_dashboard:
        if latest is None:
            st.info("Click **Generate Next Reading** in the sidebar to start the simulation.")
            return

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

                # ----------------------------------------------------
                # Real Weather Cross-Check — is this a sensor fault or
                # an actual weather event? Compares the flagged sensor
                # against live real-world weather for that city.
                # ----------------------------------------------------
                st.markdown("**🌦️ Real Weather Cross-Check**")
                real_weather = fetch_real_weather_cached(st.session_state.selected_station)
                if real_weather is None:
                    st.caption(
                        "⚠️ Live weather comparison unavailable right now "
                        "(no API key configured, or the weather service "
                        "couldn't be reached)."
                    )
                elif latest["is_multi_parameter"] or not latest["suspicious_param"]:
                    st.caption(
                        f"Multiple sensors flagged together — currently in "
                        f"**{real_weather['city']}** it's **{real_weather['description']}**, "
                        f"{real_weather['temperature']:.1f}°C. Cross-check works best for a "
                        f"single flagged sensor; please verify this one manually."
                    )
                else:
                    verdict = classify_anomaly_source(
                        latest["suspicious_param"], latest[latest["suspicious_param"]], real_weather
                    )
                    real_val = real_weather[latest["suspicious_param"]]
                    unit = SensorSimulator.units()[latest["suspicious_param"]]
                    if verdict == "matches_real_weather":
                        st.success(
                            f"🌍 **Matches Real Weather** — {real_weather['city']} is currently "
                            f"reporting {PARAM_LABELS[latest['suspicious_param']]} of "
                            f"{format_value(latest['suspicious_param'], real_val)} {unit} "
                            f"({real_weather['description']}), close to our sensor's reading. "
                            f"This may be a genuine weather event, not a fault."
                        )
                    elif verdict == "likely_sensor_fault":
                        st.warning(
                            f"🔧 **Likely Sensor Fault** — {real_weather['city']}'s real "
                            f"{PARAM_LABELS[latest['suspicious_param']]} is "
                            f"{format_value(latest['suspicious_param'], real_val)} {unit} "
                            f"({real_weather['description']}), which doesn't match our sensor's "
                            f"reading. The sensor is likely malfunctioning rather than reporting "
                            f"a real event."
                        )
                    else:
                        st.caption("Not enough data to compare for this parameter.")
                    st.caption(
                        "Best-effort automated cross-check against live public weather data — "
                        "not a certified diagnosis; a field technician should still confirm."
                    )

                # ----------------------------------------------------
                # Feature Importance — which sensor drove this anomaly
                # ----------------------------------------------------
                st.markdown("**🔍 Sensor Contribution to This Anomaly**")
                z_scores = {}
                for p in PARAMS:
                    stats = detector.feature_stats[p]
                    std = stats["std"] if stats["std"] > 1e-6 else 1e-6
                    z_scores[p] = abs((latest[p] - stats["mean"]) / std)
                total_z = sum(z_scores.values()) or 1e-6
                contribution = {p: (z / total_z) * 100 for p, z in z_scores.items()}
                sorted_params = sorted(contribution, key=lambda p: contribution[p], reverse=True)

                fi_fig = go.Figure(go.Bar(
                    x=[contribution[p] for p in sorted_params],
                    y=[PARAM_LABELS[p] for p in sorted_params],
                    orientation="h",
                    text=[f"{contribution[p]:.0f}%" for p in sorted_params],
                    textposition="outside",
                    marker_color=[
                        "#dc2626" if p == latest["suspicious_param"] else "#93c5fd"
                        for p in sorted_params
                    ],
                ))
                fi_fig.update_layout(
                    height=220, margin=dict(t=10, b=10, l=10, r=30),
                    xaxis_title="Contribution to anomaly (%)",
                    xaxis=dict(range=[0, max(contribution.values()) * 1.25]),
                )
                st.plotly_chart(fi_fig, use_container_width=True)
                st.caption(
                    "Based on how many standard deviations each sensor sits from its "
                    "own learned normal range — the sensor with the largest share is "
                    "reported as the likely cause."
                )

            # --------------------------------------------------------
            # Alert Engine — simulated notification preview
            # --------------------------------------------------------
            if station["notification_log"]:
                latest_notif = station["notification_log"][-1]
                sev = latest_notif["email"]["severity"]
                st.markdown("**📤 Alert Engine — Notification Sent**")
                st.caption(
                    "Prototype notice: no real email/SMS is sent — this shows "
                    "what the Alert Engine would dispatch to a field "
                    "technician in a production deployment."
                )
                tab_email, tab_sms = st.tabs(["📧 Email Preview", "📱 SMS Preview"])
                with tab_email:
                    em = latest_notif["email"]
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e2e8f0;border-radius:8px;
                        padding:14px 16px;background-color:#ffffff;">
                            <div style="color:#64748b;font-size:0.85rem;">To: {em['to']}</div>
                            <div style="font-weight:700;color:#0f172a;margin:4px 0;">
                                {em['subject']}
                                <span class="status-badge" style="background-color:{sev['color']};
                                margin-left:8px;">{sev['emoji']} {sev['label']}</span>
                            </div>
                            <pre style="white-space:pre-wrap;font-family:inherit;
                            color:#334155;font-size:0.9rem;margin:0;">{em['body']}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with tab_sms:
                    sm = latest_notif["sms"]
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e2e8f0;border-radius:8px;
                        padding:14px 16px;background-color:#ffffff;max-width:340px;">
                            <div style="color:#64748b;font-size:0.85rem;">To: {sm['to']}</div>
                            <div style="background-color:#f1f5f9;border-radius:8px;
                            padding:10px 12px;margin-top:6px;color:#0f172a;">{sm['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ------------------------------------------------------------
        # Anomaly history table
        # ------------------------------------------------------------
        st.subheader("🗂️ Anomaly History")
        table = history_to_display_table(station["anomaly_log"])
        if len(table):
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.caption("No anomalies logged yet. Use the sidebar to inject one.")

        if station["notification_log"]:
            with st.expander(f"📤 Notification Log ({len(station['notification_log'])} alerts sent)"):
                for n in reversed(station["notification_log"][-20:]):
                    sev = n["email"]["severity"]
                    st.markdown(
                        f"{sev['emoji']} **{n['time'].strftime('%H:%M:%S')}** — "
                        f"{n['email']['subject']} *(email + SMS)*"
                    )


live_dashboard_fragment()

# ===========================================================================
# TAB 2 — SYSTEM ARCHITECTURE (static — no need to auto-refresh)
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

# ===========================================================================
# TAB 3 — MODEL PERFORMANCE (static — no need to auto-refresh)
# ===========================================================================
with tab_metrics:
    st.subheader("📐 Model Performance")
    eval_path = os.path.join(BASE_DIR, "ml", "eval_results.json")

    if not os.path.exists(eval_path):
        st.warning(
            "No evaluation results found. Run `python ml/evaluate_model.py` "
            "from the project root, then refresh this page."
        )
    else:
        import json
        with open(eval_path) as f:
            ev = json.load(f)

        st.caption(
            f"Evaluated on a held-out synthetic test set of "
            f"**{ev['test_set_size']} readings** "
            f"({ev['normal_count']} normal, {ev['anomalous_count']} labelled "
            f"anomalous) — generated with a different random seed than the "
            f"training data, so the model never saw these exact readings."
        )

        m = ev["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m['precision']*100:.1f}%",
                   help="Of readings flagged as anomalies, how many actually were.")
        c2.metric("Recall", f"{m['recall']*100:.1f}%",
                   help="Of all true anomalies, how many the system caught.")
        c3.metric("F1 Score", f"{m['f1_score']*100:.1f}%",
                   help="Balance between precision and recall.")
        c4.metric("Accuracy", f"{m['accuracy']*100:.1f}%",
                   help="Overall correct classification rate.")

        st.write("")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Confusion Matrix**")
            cm = ev["confusion_matrix"]
            cm_fig = go.Figure(data=go.Heatmap(
                z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                texttemplate="%{text}",
                textfont={"size": 18},
                colorscale=[[0, "#dbeafe"], [1, "#2563eb"]],
                showscale=False,
            ))
            cm_fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(cm_fig, use_container_width=True)

        with col_b:
            st.markdown("**Anomaly Score Distribution**")
            sd = ev["score_distribution"]
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=sd["normal"]["values"], name="Normal readings",
                marker_color="#16a34a", opacity=0.7, nbinsx=20,
            ))
            hist_fig.add_trace(go.Histogram(
                x=sd["anomalous"]["values"], name="Anomalous readings",
                marker_color="#dc2626", opacity=0.7, nbinsx=20,
            ))
            hist_fig.update_layout(
                barmode="overlay", height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Anomaly Score (0-100)", yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        st.write("")
        st.markdown("**Parameter Identification Accuracy**")
        st.caption(
            "When an anomaly was correctly detected, how often the system "
            "pointed at the *right* sensor as the cause."
        )
        id_acc = ev["per_parameter_identification_accuracy"]
        cols = st.columns(len(id_acc))
        for col, (param, acc) in zip(cols, id_acc.items()):
            col.metric(PARAM_LABELS.get(param, param), f"{acc*100:.0f}%" if acc is not None else "—")

        st.write("")
        with st.expander("Model configuration"):
            info = ev["model_info"]
            st.markdown(
                f"""
- **Algorithm:** {info['algorithm']}
- **Trees (n_estimators):** {info['n_estimators']}
- **Contamination setting:** {info['contamination']}
- **Features used:** {', '.join(info['features_used'])}
- **Training samples:** {info['training_samples']}
                """
            )
        st.caption(
            "To regenerate these results after retraining the model, run "
            "`python ml/evaluate_model.py` from the project root."
        )
