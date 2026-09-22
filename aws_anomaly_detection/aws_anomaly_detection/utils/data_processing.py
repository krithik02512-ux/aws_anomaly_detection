"""
data_processing.py
=====================
Small, dependency-light helper functions used by the Streamlit dashboard
(app.py) to keep app.py focused on layout/UI rather than logic.
"""

from typing import List, Optional
import pandas as pd


def compute_trend(history: pd.DataFrame, param: str, lookback: int = 5) -> str:
    """
    Returns a simple trend arrow for a parameter based on the last
    `lookback` readings: "up", "down", or "flat".
    """
    if history is None or len(history) < 2:
        return "flat"

    recent = history[param].tail(lookback)
    if len(recent) < 2:
        return "flat"

    delta = recent.iloc[-1] - recent.iloc[0]
    # small dead-zone so tiny noise doesn't flip the arrow every tick
    threshold = recent.std() * 0.3 if recent.std() > 0 else 0.01
    if delta > threshold:
        return "up"
    elif delta < -threshold:
        return "down"
    return "flat"


TREND_ARROWS = {"up": "▲", "down": "▼", "flat": "▶"}


def trend_arrow(trend: str) -> str:
    return TREND_ARROWS.get(trend, "▶")


def param_status(param: str, suspicious_param: Optional[str], is_anomaly: bool,
                  z_score: float, notable_threshold: float = 2.0) -> str:
    """
    Per-card status label for an individual sensor:
      - "anomaly": this exact sensor is the one the ML model flagged
      - "warning": not the primary flagged sensor, but still notably
                    off from its learned normal range
      - "normal" : within expected range
    """
    if is_anomaly and suspicious_param == param:
        return "anomaly"
    if abs(z_score) >= notable_threshold:
        return "warning"
    return "normal"


STATUS_COLORS = {
    "normal": "#16a34a",   # green
    "warning": "#d97706",  # amber
    "anomaly": "#dc2626",  # red
}

STATUS_LABELS = {
    "normal": "Normal",
    "warning": "Warning",
    "anomaly": "Anomaly",
}


def format_value(param: str, value: float) -> str:
    decimals = 1 if param in ("temperature", "humidity", "wind_speed", "rainfall") else 2
    return f"{value:.{decimals}f}"


def history_to_display_table(anomaly_log: List[dict]) -> pd.DataFrame:
    """Builds the 'Time | Parameter | Value | Anomaly Score | Status' table
    from a list of logged anomaly events (most recent first)."""
    if not anomaly_log:
        return pd.DataFrame(
            columns=["Time", "Parameter", "Value", "Anomaly Score", "Status"]
        )

    rows = []
    for entry in reversed(anomaly_log):  # most recent first
        rows.append(
            {
                "Time": entry["time"].strftime("%H:%M:%S"),
                "Parameter": entry["parameter"],
                "Value": entry["value"],
                "Anomaly Score": entry["score"],
                "Status": entry["status"],
            }
        )
    return pd.DataFrame(rows)
