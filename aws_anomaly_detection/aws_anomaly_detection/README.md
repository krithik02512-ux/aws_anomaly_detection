# AI-Powered Automatic Weather Station (AWS) Monitoring
### Intelligent Multi-Sensor Anomaly Detection — Working Prototype

> ⚠️ **Note on scope:** The official problem statement for this competition only
> provides the title *"AI/ML-Based Intelligent Anomaly Detection for Automatic
> Weather Stations (AWS)."* Everything below — the specific features, ML
> approach, architecture and UI — is **our team's proposed solution**, not an
> official requirement.

---

## 1. Project Overview

Automatic Weather Stations (AWS) run unattended for long periods and report
readings (temperature, humidity, pressure, wind speed, rainfall) with no
human nearby to notice if a sensor starts misbehaving. This prototype is a
web-based monitoring system that:

1. Simulates live AWS sensor data (since we don't have physical hardware yet).
2. Feeds every new reading through a trained **unsupervised ML model**
   (Isolation Forest) that has learned what a "normal" combination of the
   5 sensors looks like.
3. Flags readings that don't fit the learned normal pattern.
4. Identifies **which sensor** is most likely responsible for the anomaly.
5. Displays everything on a live monitoring dashboard, with an explainable
   alert and history log.

This is **not** primarily a weather-forecasting app — the objective is
anomaly detection, not predicting tomorrow's weather.

## 2. Problem Definition

A single out-of-range threshold on one sensor (e.g. "alert if temperature
> 50°C") misses a large and important class of problems:

- It can't tell when a sensor looks wrong **relative to the other sensors**
  at that same moment, rather than relative to a fixed number.
- It can't distinguish "this looks like a plausible unusual weather event"
  from "this looks like an isolated sensor glitch."
- It gives no sense of *how* abnormal a reading is, just a binary yes/no.

## 3. Proposed Solution

Use an unsupervised anomaly-detection model trained on the joint behaviour
of all 5 sensors together, so the system can catch:

- **Single-sensor faults**: one sensor value far outside anything ever seen,
  while the rest of the station reports normally.
- **Unusual multi-parameter conditions**: several sensors shifting together
  in a way that's statistically unusual, which the system flags for human
  verification rather than confidently calling it a "fault."

## 4. Key Differentiators

This prototype is **not** claiming to invent AI-based weather anomaly
detection. What differentiates our implementation:

1. **Multi-sensor analysis** — all 5 parameters are analysed jointly, not
   one at a time.
2. **Anomaly scoring** — a continuous 0–100 score, not just a binary flag.
3. **Suspicious-parameter identification** — the system points at *which*
   sensor is most likely responsible.
4. **Explainable alerts** — every alert includes a plain-language reason
   (e.g. "Temperature reading significantly deviates X standard deviations
   above the learned normal average").
5. **Interactive anomaly injection** — live, on-demand demonstration
   controls for a competition demo.
6. **Clean real-time dashboard** — built to be understood by judges within
   about 10 seconds of looking at it.

## 5. System Architecture

```
Virtual Sensors
      ↓
Data Acquisition
      ↓
Data Preprocessing
      ↓
ML Anomaly Detection  (Isolation Forest)
      ↓
Anomaly Score (0–100)
      ↓
Parameter Identification  (z-score analysis)
      ↓
Alert Engine
      ↓
Dashboard
```

This same diagram, plus a plain-language explanation of the ML model, is
also shown inside the app on the **"System Architecture"** tab.

## 6. Technology Stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| Dashboard/UI   | Streamlit                            |
| Charts         | Plotly                               |
| ML             | scikit-learn (Isolation Forest)      |
| Data handling  | Pandas, NumPy                        |
| Data storage   | CSV (synthetic dataset)              |
| Model storage  | joblib (`ml/model.pkl`)              |

No React/Node/databases/Docker/cloud infrastructure — the goal is a
prototype that runs on a normal student laptop with a single command.

## 7. Project Structure

```
aws_anomaly_detection/
│
├── app.py                      # Streamlit dashboard (main entry point)
├── requirements.txt
│
├── data/
│   └── weather_data.csv        # Synthetic "normal" training dataset
│
├── ml/
│   ├── train_model.py          # Generates data + trains Isolation Forest
│   ├── anomaly_detector.py     # Loads model, analyses readings, explains results
│   ├── evaluate_model.py       # Builds a labelled test set + computes metrics
│   ├── model.pkl               # Trained model + scaler + stats (generated)
│   └── eval_results.json       # Precision/recall/F1/confusion matrix (generated)
│
├── simulation/
│   └── sensor_simulator.py     # Virtual AWS sensor data generator
│
├── utils/
│   └── data_processing.py      # Dashboard helper functions (trend, formatting)
│
└── README.md
```

## 8. Installation Instructions

**Requirements:** Python 3.9+ and internet access (only needed once, to
install the packages below).

```bash
# 1. Move into the project folder
cd aws_anomaly_detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 9. How to Run

```bash
# Step 1: Train the model (generates data/weather_data.csv and ml/model.pkl)
python ml/train_model.py

# Step 2: Evaluate the model (generates ml/eval_results.json for the
#          "Model Performance" tab in the dashboard)
python ml/evaluate_model.py

# Step 3: Launch the dashboard
streamlit run app.py

# Step 4: Open the URL Streamlit prints (usually http://localhost:8501)
```

> The app will also auto-generate `ml/model.pkl` the first time via
> `train_model.py` — make sure Step 1 has been run at least once before
> Step 2, or `app.py` will raise a clear error telling you to run it.

## 10. How the ML Model Works

- **Training data**: `ml/train_model.py` generates ~5,000 synthetic
  *normal* readings with realistic relationships (daily temperature cycle,
  humidity roughly inverse to temperature, slowly-drifting pressure, a
  skewed wind-speed distribution, and bursty rainfall).
- **Model**: an `IsolationForest` (200 trees) is trained on the 5
  standardised features. Isolation Forest works by randomly partitioning
  the data — points that sit apart from the rest get isolated into their
  own partition in very few splits, while typical points take many splits.
  That "ease of isolation" becomes the anomaly score.
- **Anomaly score**: the model's raw score is rescaled to an intuitive
  0–100 scale (0 = typical, 100 = extremely abnormal), centred on the
  model's own decision boundary.
- **Suspicious-parameter identification**: once a reading is flagged, the
  system computes a z-score for each sensor against its learned normal
  mean/std, and reports the sensor with the largest deviation as the
  likely cause. If several sensors are notably off at once, it reports
  *"unusual multi-parameter condition — requires verification"* instead of
  blaming a single sensor.
- **Statistical safety net**: Isolation Forest's split thresholds are
  bounded to the range of values it saw during training, so in rare cases
  a value that is wildly outside anything ever seen (e.g. a stuck sensor
  reporting 65°C) might not get "isolated" quickly by the trees alone. To
  keep detection reliable, the system also flags a reading if **any single
  sensor** is more than 3.5 standard deviations from its learned normal
  mean, regardless of what the ML model alone decided. This is a simple,
  transparent rule layered **on top of** the ML model — not a replacement
  for it — and it's easy to explain to judges: *"the ML model catches
  unusual combinations; the statistical guardrail catches single-sensor
  values so extreme the model never saw anything like them during
  training."*

## 11. How to Demonstrate Anomalies

The sidebar has one button per sensor:

```
[🌡️ Temperature]  [💧 Humidity]  [🧭 Pressure]  [💨 Wind Speed]  [🌧️ Rainfall]
[✅ Reset to Normal]
```

Recommended 2–3 minute demo flow:

1. Open the dashboard — all 5 sensor cards show 🟢 **Normal** status.
2. Point out the live trend graphs and explain the pipeline (see the
   System Architecture tab).
3. Click **Inject Temperature Anomaly**.
4. Point out: sensor card turns red, the system banner switches to
   🔴 **ANOMALY DETECTED**, the graph marks the anomalous point, and the
   alert panel shows the detection time, value, anomaly score and a
   plain-language explanation.
5. Click **Reset to Normal** — banner returns to 🟢 within a couple of
   readings.
6. Repeat with a different sensor (e.g. **Inject Wind Anomaly**) to show
   the system correctly identifies a *different* suspicious parameter each
   time.
7. Optionally, toggle **Live auto-refresh** to show the dashboard updating
   continuously on its own.

## 12. Limitations

This is an engineering-fair **prototype**, and we are explicit about what
it is *not*:

- It uses **simulated/synthetic AWS data**, not certified meteorological
  hardware or live weather data.
- It does **not** guarantee it can always correctly distinguish a genuine
  sensor fault from an unusual-but-real weather event — where several
  sensors move together plausibly, it deliberately reports "requires
  verification" rather than a confident diagnosis.
- It is **not** a weather-forecasting or disaster-prediction system.
- It is **not** production-deployment ready (no real sensor integration,
  authentication, persistent database, or hardware failover).
- Isolation Forest is trained only on the synthetic distribution we
  generated; on real AWS field data the model would need to be retrained
  on real historical sensor logs before being trusted.

## 13. Future Improvements

- Replace the virtual simulator with real AWS hardware / IoT sensor feed.
- Retrain on real historical station data instead of synthetic data.
- Add more sensors (e.g. solar radiation, soil moisture) as additional
  features.
- Compare Isolation Forest against other unsupervised approaches
  (One-Class SVM, Autoencoders, Local Outlier Factor) for accuracy.
- Add persistent storage (database) for long-term anomaly history and
  reporting.
- Add SMS/email alerting for field technicians.
- Add per-station model calibration if deployed across multiple AWS units
  with different local baselines.

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Trained model not found at ml/model.pkl` | Run `python ml/train_model.py` first. |
| `ModuleNotFoundError: No module named 'streamlit'` (or pandas/sklearn/plotly) | Run `pip install -r requirements.txt` inside your virtual environment. |
| Dashboard opens but looks empty | Click **Generate Next Reading** in the sidebar once. |
| Port already in use | Run `streamlit run app.py --server.port 8502` (or any free port). |
| Changes to `app.py` don't show up | Streamlit auto-reloads on save; if not, refresh the browser tab or restart `streamlit run app.py`. |

---

*Built as a functional engineering prototype: functionality first, then
simplicity, then a professional UI — in that order of priority.*
reportlab
