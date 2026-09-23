"""
evaluate_model.py
====================
Builds a labelled TEST set (separate from training) and evaluates the
trained anomaly detector against it, saving the results to
ml/eval_results.json so the Streamlit dashboard can display a
"Model Performance" page without recomputing this on every run.

Since Isolation Forest is unsupervised (trained only on normal data, with
no labels), it cannot be evaluated the way a supervised classifier is.
Instead we build a SEPARATE synthetic test set where we DO know the
ground truth:
    - "normal" rows: generated the same way as the training data, but
      with a different random seed (held-out, never seen during training).
    - "anomalous" rows: built by spiking one parameter far outside its
      normal range (the same style of spike used by the live demo's
      "Inject Anomaly" buttons), so we know for certain these SHOULD be
      flagged.

This lets us report standard classification metrics (precision, recall,
F1, accuracy, confusion matrix) even though the model itself never saw
any labels during training.

Run with:  python ml/evaluate_model.py   (from the project root)
       or: python evaluate_model.py      (from inside ml/)
"""

import json
import os
import sys
import random

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from anomaly_detector import AnomalyDetector          # noqa: E402
from train_model import generate_normal_weather_data, FEATURE_NAMES  # noqa: E402

TEST_SEED = 2024  # different from training's RANDOM_SEED=42 -> true held-out data


def build_anomalous_samples(rng: random.Random, n_per_param: int = 20):
    """Builds labelled anomalous rows by spiking one parameter at a time,
    mirroring the live demo's anomaly-injection logic."""
    baseline = {
        "temperature": {"mean": 28.0, "std": 4.3},
        "humidity": {"mean": 75.0, "std": 4.5},
        "pressure": {"mean": 1010.0, "std": 1.4},
        "wind_speed": {"mean": 12.0, "std": 8.3},
        "rainfall": {"mean": 0.25, "std": 1.4},
    }
    rows = []
    for param in FEATURE_NAMES:
        for _ in range(n_per_param):
            row = {p: rng.gauss(baseline[p]["mean"], baseline[p]["std"]) for p in FEATURE_NAMES}
            cfg = baseline[param]
            direction = 1 if param in ("wind_speed", "rainfall") else rng.choice([1, -1])
            magnitude = rng.uniform(6, 10) * cfg["std"]
            row[param] = cfg["mean"] + direction * magnitude
            # keep physically plausible bounds, same as sensor_simulator.py
            row["humidity"] = max(0.0, min(100.0, row["humidity"]))
            row["wind_speed"] = max(0.0, row["wind_speed"])
            row["rainfall"] = max(0.0, row["rainfall"])
            rows.append({"features": row, "true_label": 1, "true_param": param})
    return rows


def evaluate():
    detector = AnomalyDetector()
    rng = random.Random(TEST_SEED)

    # --- Held-out NORMAL test rows (different seed from training) ---
    np.random.seed(TEST_SEED)
    normal_df = generate_normal_weather_data(n_samples=500)
    normal_rows = [
        {"features": {f: float(row[f]) for f in FEATURE_NAMES}, "true_label": 0, "true_param": None}
        for _, row in normal_df.iterrows()
    ]

    # --- Labelled ANOMALOUS test rows ---
    anomalous_rows = build_anomalous_samples(rng, n_per_param=20)

    all_rows = normal_rows + anomalous_rows

    tp = fp = tn = fn = 0
    normal_scores, anomaly_scores = [], []
    per_param_correct = {p: 0 for p in FEATURE_NAMES}
    per_param_total = {p: 0 for p in FEATURE_NAMES}

    for row in all_rows:
        result = detector.analyze(row["features"])
        predicted = 1 if result.is_anomaly else 0
        true = row["true_label"]

        if true == 1:
            anomaly_scores.append(result.anomaly_score)
        else:
            normal_scores.append(result.anomaly_score)

        if true == 1 and predicted == 1:
            tp += 1
        elif true == 0 and predicted == 1:
            fp += 1
        elif true == 0 and predicted == 0:
            tn += 1
        elif true == 1 and predicted == 0:
            fn += 1

        if row["true_param"] is not None:
            per_param_total[row["true_param"]] += 1
            if predicted == 1 and (result.suspicious_param == row["true_param"] or result.is_multi_parameter):
                per_param_correct[row["true_param"]] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    results = {
        "test_set_size": len(all_rows),
        "normal_count": len(normal_rows),
        "anomalous_count": len(anomalous_rows),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "score_distribution": {
            "normal": {
                "mean": round(float(np.mean(normal_scores)), 2),
                "median": round(float(np.median(normal_scores)), 2),
                "max": round(float(np.max(normal_scores)), 2),
                "values": [round(s, 1) for s in normal_scores],
            },
            "anomalous": {
                "mean": round(float(np.mean(anomaly_scores)), 2),
                "median": round(float(np.median(anomaly_scores)), 2),
                "min": round(float(np.min(anomaly_scores)), 2),
                "values": [round(s, 1) for s in anomaly_scores],
            },
        },
        "per_parameter_identification_accuracy": {
            p: round(per_param_correct[p] / per_param_total[p], 4) if per_param_total[p] else None
            for p in FEATURE_NAMES
        },
        "model_info": {
            "algorithm": "Isolation Forest",
            "n_estimators": detector.model.n_estimators,
            "contamination": detector.model.contamination,
            "features_used": FEATURE_NAMES,
            "training_samples": 5000,
        },
    }

    out_path = os.path.join(BASE_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate_model] Test set: {len(all_rows)} rows "
          f"({len(normal_rows)} normal, {len(anomalous_rows)} anomalous)")
    print(f"[evaluate_model] Precision: {precision:.3f}  Recall: {recall:.3f}  "
          f"F1: {f1:.3f}  Accuracy: {accuracy:.3f}")
    print(f"[evaluate_model] Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[evaluate_model] Results saved to {out_path}")


if __name__ == "__main__":
    evaluate()
