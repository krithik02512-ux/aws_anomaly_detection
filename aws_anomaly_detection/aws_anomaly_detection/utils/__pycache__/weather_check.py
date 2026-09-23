"""
utils/weather_check.py
========================
Cross-checks a flagged anomaly against REAL current weather (via the
OpenWeatherMap API) for that station's city, to help distinguish:

    - "Likely Sensor Fault"     -> real weather is normal, only our sensor
                                    is showing something unusual
    - "Matches Real Weather"    -> real weather independently confirms the
                                    unusual reading (e.g. an actual storm)
    - "Uncertain"               -> API unavailable / no data to compare

This is a best-effort heuristic cross-check, NOT a certified diagnosis.
"""

import requests

# Approximate coordinates for each virtual AWS station's city.
STATION_COORDS = {
    "AWS-STN-01": {"lat": 13.0827, "lon": 80.2707, "city": "Chennai"},
    "AWS-STN-02": {"lat": 11.0168, "lon": 76.9558, "city": "Coimbatore"},
    "AWS-STN-03": {"lat": 9.9252, "lon": 78.1198, "city": "Madurai"},
}

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# How far off (relative %) a sensor reading can be from the real API
# reading before we call it a mismatch. Chosen loosely since a single
# ground sensor and a city-wide weather API reading will never match
# exactly, even when both are "correct".
TOLERANCE_PCT = {
    "temperature": 15,
    "humidity": 25,
    "pressure": 3,
    "wind_speed": 60,   # gusty/local wind varies a lot from city-average API data
    "rainfall": 100,    # rainfall is extremely local; API often reads 0 nearby
}


def get_real_weather(station_id: str, api_key: str):
    """Fetches current real weather for a station's city.
    Returns a dict of {temperature, humidity, pressure, wind_speed,
    rainfall, description} on success, or None on any failure
    (missing key, network error, bad response, unknown station)."""
    coords = STATION_COORDS.get(station_id)
    if not coords or not api_key:
        return None
    try:
        resp = requests.get(
            OPENWEATHER_URL,
            params={
                "lat": coords["lat"],
                "lon": coords["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=6,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "city": coords["city"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"].get("speed", 0.0),
            "rainfall": data.get("rain", {}).get("1h", 0.0),
            "description": data["weather"][0]["description"].title(),
        }
    except Exception:
        return None


def classify_anomaly_source(suspicious_param: str, sensor_value: float, real_weather: dict):
    """Compares the flagged sensor's value against the real current value
    for that same parameter, and returns one of:
        "likely_sensor_fault", "matches_real_weather", "uncertain"
    """
    if not real_weather or not suspicious_param or suspicious_param not in real_weather:
        return "uncertain"

    real_value = real_weather[suspicious_param]
    tolerance = TOLERANCE_PCT.get(suspicious_param, 20)

    # Special-case rainfall/pressure edge values near zero to avoid
    # divide-by-zero / meaningless percentage swings.
    denom = max(abs(real_value), 1.0)
    diff_pct = abs(sensor_value - real_value) / denom * 100

    if diff_pct <= tolerance:
        return "matches_real_weather"
    return "likely_sensor_fault"
