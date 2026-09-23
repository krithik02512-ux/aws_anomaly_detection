"""
alert_engine.py
==================
Turns a detected anomaly into a notification that WOULD be sent to a field
technician in a production deployment (email + SMS style previews).

This prototype does NOT actually send real emails/SMS (no mail server or
SMS gateway credentials are configured) - it simulates what the Alert
Engine stage of the pipeline would produce, so the dashboard can show a
realistic "this is what the technician receives" preview. See
`send_real_email()` at the bottom for how this would be wired up to a
real SMTP server in production (kept separate and unused by default).
"""

from datetime import datetime
from typing import Dict, Optional

# In a real deployment these would come from a config file / environment
# variables per-station, not be hard-coded.
DEFAULT_RECIPIENT_EMAIL = "field.technician@awsstation.local"
DEFAULT_RECIPIENT_PHONE = "+91-XXXXXXXXXX"
STATION_ID = "AWS-STN-01"


def severity_label(anomaly_score: float) -> Dict[str, str]:
    """Maps the 0-100 anomaly score to a human severity tier, used to
    decide how urgently a notification should be treated."""
    if anomaly_score >= 85:
        return {"label": "CRITICAL", "color": "#dc2626", "emoji": "🔴"}
    elif anomaly_score >= 60:
        return {"label": "MODERATE", "color": "#d97706", "emoji": "🟠"}
    else:
        return {"label": "MINOR", "color": "#ca8a04", "emoji": "🟡"}


def build_email_notification(
    parameter: Optional[str],
    parameter_label: str,
    value_display: str,
    anomaly_score: float,
    explanation: str,
    detected_at: datetime,
    is_multi_parameter: bool,
    recipient: str = DEFAULT_RECIPIENT_EMAIL,
) -> Dict[str, str]:
    """Builds the subject/body of the email a technician would receive."""
    sev = severity_label(anomaly_score)
    subject = (
        f"[{sev['label']}] AWS Anomaly Alert - {STATION_ID} - "
        f"{'Multiple Parameters' if is_multi_parameter else parameter_label}"
    )
    body = (
        f"Automatic Weather Station Monitoring System\n"
        f"Station: {STATION_ID}\n"
        f"Detected at: {detected_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Severity: {sev['label']}\n"
        f"Anomaly Score: {anomaly_score:.1f} / 100\n\n"
        f"Parameter: {'Multiple parameters (requires verification)' if is_multi_parameter else parameter_label}\n"
        f"Current Value: {value_display}\n\n"
        f"Details:\n{explanation}\n\n"
        f"Please verify the station and sensor readings at your earliest "
        f"convenience.\n\n"
        f"-- This is an automated message from the AWS Anomaly Detection System --"
    )
    return {"to": recipient, "subject": subject, "body": body, "severity": sev}


def build_sms_notification(
    parameter_label: str,
    anomaly_score: float,
    is_multi_parameter: bool,
    recipient: str = DEFAULT_RECIPIENT_PHONE,
) -> Dict[str, str]:
    """Builds a short SMS-style alert (kept under ~160 characters)."""
    sev = severity_label(anomaly_score)
    param_text = "Multiple params" if is_multi_parameter else parameter_label
    text = (
        f"[{STATION_ID}] {sev['label']} anomaly: {param_text}. "
        f"Score {anomaly_score:.0f}/100. Check dashboard."
    )
    return {"to": recipient, "text": text, "severity": sev}


# ---------------------------------------------------------------------------
# REFERENCE ONLY — not called anywhere in this prototype. Shows how a real
# deployment would wire this up to an actual mail server using Python's
# built-in smtplib, once real SMTP credentials are available (e.g. via
# Streamlit secrets: st.secrets["smtp_user"], st.secrets["smtp_password"]).
# ---------------------------------------------------------------------------
def send_real_email(smtp_host, smtp_port, smtp_user, smtp_password, to_addr, subject, body):
    """Not used in the prototype - reference implementation only."""
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [to_addr], msg.as_string())
