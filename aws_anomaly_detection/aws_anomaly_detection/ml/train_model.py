"""
train_model.py
================
Generates a realistic synthetic dataset that represents NORMAL Automatic
Weather Station (AWS) sensor behaviour, and trains an unsupervised
Isolation Forest model on it.

Why Isolation Forest?
----------------------
- We do NOT have labelled "anomaly / not anomaly" data from a real AWS,
  so a supervised model (which needs labels) is not appropriate.
- Isolation Forest is an unsupervised algorithm that learns what "normal"
  combinations of sensor readings look like, and isolates points that are
  easy to separate from the rest (few random splits needed) -> those are
  flagged as anomalies.
- It works well with small/medium tabular data, is fast to train, and is
  simple enough for a 2nd-year engineering student to explain in a demo:
  "the model learned the normal pattern of 5 sensors together, and flags
  any new reading that doesn't fit that learned pattern."

Output artefacts (saved to ml/model.pkl):
    - model            : trained IsolationForest
    - scaler           : StandardScaler fit on the training features
    - feature_names    : ordered list of feature names used by the model
    - feature_stats    : per-feature mean/std/min/max (used later to
                          explain WHICH parameter looks suspicious)
    - score_bounds     : min/max of the raw decision_function scores seen
                          during training, used to rescale the anomaly
                          score into an easy-to-read 0-100 range.

Also writes: data/weather_data.csv (the synthetic training dataset).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_NAMES = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]


def generate_normal_weather_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates realistic-looking NORMAL weather-station data.

    The physical relationships modelled (kept intentionally simple):
    - Temperature follows a daily sinusoidal cycle (cooler at night,
      warmer in the afternoon) + small random noise.
    - Humidity is loosely INVERSE to temperature (hot afternoons tend to
      be less humid) + noise, clipped to a realistic band.
    - Pressure stays close to a stable baseline (~1010 hPa) with slow,
      small drifts - real atmospheric pressure rarely jumps quickly.
    - Wind speed is drawn from a skewed (exponential-like) distribution
      since calm conditions are more common than strong wind.
    - Rainfall is mostly 0, with occasional rain "events" - rainfall is
      not continuous, it happens in bursts.

    These relationships are what the Isolation Forest implicitly learns.
    A reading that breaks this learned joint pattern (e.g. temperature
    spikes while everything else stays put) will look "isolated" and get
    flagged.
    """
    t = np.arange(n_samples)

    # --- Temperature: daily cycle between ~22C and ~34C + noise ---
    daily_cycle = np.sin(2 * np.pi * (t % 288) / 288)  # 288 = readings/day if 5-min interval
    temperature = 28 + 6 * daily_cycle + np.random.normal(0, 0.8, n_samples)

    # --- Humidity: inversely related to temperature, clipped 30-95% ---
    humidity = 75 - 4 * daily_cycle + np.random.normal(0, 3, n_samples)
    humidity = np.clip(humidity, 30, 95)

    # --- Pressure: stable baseline with slow random-walk drift ---
    pressure_drift = np.cumsum(np.random.normal(0, 0.05, n_samples))
    pressure_drift = np.clip(pressure_drift, -8, 8)  # keep drift bounded
    pressure = 1010 + pressure_drift + np.random.normal(0, 0.3, n_samples)

    # --- Wind speed: skewed distribution, mostly calm/light wind ---
    wind_speed = np.random.gamma(shape=2.0, scale=6.0, size=n_samples)
    wind_speed = np.clip(wind_speed, 0, 45)

    # --- Rainfall: mostly dry, occasional rain bursts ---
    rain_event = np.random.random(n_samples) < 0.06  # ~6% of readings have rain
    rainfall = np.where(
        rain_event,
        np.random.exponential(scale=4.0, size=n_samples),
        0.0,
    )
    rainfall = np.clip(rainfall, 0, 40)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                end=pd.Timestamp.now(), periods=n_samples, freq="5min"
            ),
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 2),
            "pressure": np.round(pressure, 2),
            "wind_speed": np.round(wind_speed, 2),
            "rainfall": np.round(rainfall, 2),
        }
    )
    return df


def train_and_save_model(df: pd.DataFrame, model_path: str):
    """Trains the IsolationForest + StandardScaler and saves everything
    needed for inference (including stats used for explainability)."""

    X = df[FEATURE_NAMES].values

    # Scale features so no single sensor (e.g. pressure ~1010) dominates
    # the distance/split calculations just because of its larger numbers.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination = expected proportion of anomalies in training data.
    # We trained on data we generated as "normal", so we set this low -
    # it just tells the model how strict its internal decision boundary
    # should be.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        max_samples="auto",
        random_state=RANDOM_SEED,
    )
    model.fit(X_scaled)

    # decision_function: higher = more "normal", lower/negative = more
    # anomalous. We record the min/max seen on training data so we can
    # later rescale any new score into an intuitive 0-100 range.
    raw_scores = model.decision_function(X_scaled)
    score_bounds = {
        "min": float(raw_scores.min()),
        "max": float(raw_scores.max()),
        "std": float(raw_scores.std()),
        "offset": float(model.offset_),  # the model's own inlier/outlier boundary
    }

    # Per-feature statistics (on RAW, unscaled values) - used later to
    # figure out WHICH sensor looks the most "off" for an anomalous
    # reading (z-score based explainability).
    feature_stats = {
        name: {
            "mean": float(df[name].mean()),
            "std": float(df[name].std()),
            "min": float(df[name].min()),
            "max": float(df[name].max()),
        }
        for name in FEATURE_NAMES
    }

    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "feature_stats": feature_stats,
        "score_bounds": score_bounds,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artefact, model_path)
    print(f"[train_model] Model saved to {model_path}")
    print(f"[train_model] Score bounds: {score_bounds}")
    print(f"[train_model] Feature stats: {feature_stats}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "weather_data.csv")
    model_path = os.path.join(base_dir, "ml", "model.pkl")

    print("[train_model] Generating synthetic NORMAL weather data...")
    dataset = generate_normal_weather_data(n_samples=5000)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    dataset.to_csv(data_path, index=False)
    print(f"[train_model] Dataset saved to {data_path} ({len(dataset)} rows)")

    print("[train_model] Training Isolation Forest model...")
    train_and_save_model(dataset, model_path)
    print("[train_model] Done.")
