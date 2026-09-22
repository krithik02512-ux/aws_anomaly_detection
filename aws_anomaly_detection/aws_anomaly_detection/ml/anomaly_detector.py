"""
anomaly_detector.py
=====================
Loads the trained Isolation Forest artefact and turns a single new
sensor reading into a full, EXPLAINABLE anomaly-detection result.

This is the "brain" of the project. Given one reading like:

    {"temperature": 65, "humidity": 70, "pressure": 1010,
     "wind_speed": 15, "rainfall": 2}

it returns a dictionary containing:
    - is_anomaly        : True/False (from the ML model)
    - anomaly_score      : 0-100 scale, higher = more abnormal
    - suspicious_param    : the single sensor most responsible, if any
    - param_deviations   : z-score of every sensor vs. learned normal stats
    - is_multi_parameter : True if several sensors are unusual together
    - explanation        : a short, human-readable sentence

Design notes for the demo / presentation:
-------------------------------------------
1. The overall "is this reading weird at all?" decision comes from the
   ML model (Isolation Forest), NOT from hand-picked thresholds.
2. Once the model says "this is unusual", we use simple, transparent
   z-scores (how many standard deviations a value is from the learned
   normal mean) purely to explain WHICH sensor is most responsible.
   This keeps the "why" understandable without hiding it inside a
   black-box model.
3. If more than one sensor is unusual at the same time, we deliberately
   avoid confidently blaming a single sensor fault - some real weather
   events (e.g. a storm front) can shift several parameters together.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import joblib
import numpy as np


# A reading is only considered "notably off" for a given sensor if its
# z-score magnitude exceeds this. Used purely for the explainability
# layer (which sensor looks suspicious), not for the core anomaly call.
Z_SCORE_NOTABLE = 2.0
# If more than one sensor crosses this z-score, we treat it as a
# possible genuine multi-parameter weather event rather than a single
# sensor fault.
MULTI_PARAM_Z_THRESHOLD = 2.0

# --- Safety-net threshold (secondary signal, NOT the main detector) ------
# Isolation Forest splits are bounded to the value range it saw during
# training, so a single sensor value that is wildly outside anything the
# model has ever seen (e.g. a stuck/faulty sensor reporting 65C) can, in
# rare cases, fail to be "isolated" quickly by the trees even though it is
# obviously implausible. To keep the system reliable, we back the ML model
# up with a simple statistical guardrail: if ANY single sensor is more than
# HARD_Z_GUARDRAIL standard deviations from its learned normal mean, the
# reading is flagged regardless of what the ML model alone decided. This is
# a deliberately simple, transparent safety net layered on top of the ML
# model - not a replacement for it. The ML model remains responsible for
# catching the harder case: several sensors shifting together in a way
# that looks unusual jointly even though no single sensor is extreme.
HARD_Z_GUARDRAIL = 3.5


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float  # 0-100, higher = more abnormal
    suspicious_param: Optional[str]
    param_deviations: Dict[str, float]  # z-scores per feature
    is_multi_parameter: bool
    explanation: str
    raw_values: Dict[str, float] = field(default_factory=dict)


class AnomalyDetector:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "ml", "model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                f"Run `python ml/train_model.py` first."
            )

        artefact = joblib.load(model_path)
        self.model = artefact["model"]
        self.scaler = artefact["scaler"]
        self.feature_names: List[str] = artefact["feature_names"]
        self.feature_stats: Dict[str, dict] = artefact["feature_stats"]
        self.score_bounds: Dict[str, float] = artefact["score_bounds"]

    # ------------------------------------------------------------------
    def _to_feature_vector(self, reading: Dict[str, float]) -> np.ndarray:
        return np.array([[reading[name] for name in self.feature_names]])

    def _z_scores(self, reading: Dict[str, float]) -> Dict[str, float]:
        z_scores = {}
        for name in self.feature_names:
            stats = self.feature_stats[name]
            std = stats["std"] if stats["std"] > 1e-6 else 1e-6
            z_scores[name] = (reading[name] - stats["mean"]) / std
        return z_scores

    def _rescale_score(self, raw_score: float) -> float:
        """
        Rescales the raw Isolation Forest decision_function output into
        an intuitive 0-100 "anomaly score" using a logistic (sigmoid)
        transform centred on the model's own inlier/outlier boundary
        (model.offset_):
            score ≈ 50  -> reading sits right on the model's decision
                            boundary
            score  < 50  -> looks normal (further below 50 = more typical)
            score  > 50  -> looks abnormal (further above 50 = more extreme)

        A plain min/max rescale over the (very narrow) range of scores
        seen on training data makes ordinary noise look artificially
        close to the anomaly boundary, so we instead scale by the
        STANDARD DEVIATION of training scores, which spreads genuinely
        typical points down near 0 and lets truly extreme points push
        all the way up toward 100.
        """
        offset = self.score_bounds.get("offset", 0.0)
        std = self.score_bounds.get("std", 0.05) or 0.05
        centred = raw_score - offset
        # logistic transform: higher raw_score (more normal) -> low score
        anomaly_score = 100.0 / (1.0 + np.exp(centred / std))
        return round(float(np.clip(anomaly_score, 0, 100)), 1)

    # ------------------------------------------------------------------
    def analyze(self, reading: Dict[str, float]) -> AnomalyResult:
        """Main entry point: takes one sensor reading, returns a full
        explainable AnomalyResult."""

        X = self._to_feature_vector(reading)
        X_scaled = self.scaler.transform(X)

        raw_prediction = self.model.predict(X_scaled)[0]  # 1 = normal, -1 = anomaly
        raw_score = self.model.decision_function(X_scaled)[0]

        z_scores = self._z_scores(reading)
        max_abs_z = max(abs(z) for z in z_scores.values())

        ml_flag = bool(raw_prediction == -1)
        guardrail_flag = max_abs_z >= HARD_Z_GUARDRAIL

        # Final decision combines both signals. The ML model is the
        # primary detector for unusual JOINT combinations of sensors;
        # the guardrail catches single-sensor values so extreme that
        # they fall outside the model's own training experience.
        is_anomaly = ml_flag or guardrail_flag

        anomaly_score = self._rescale_score(raw_score)
        if guardrail_flag:
            # Make sure the displayed score reflects how extreme the
            # reading is even in the rare case the ML score alone
            # wouldn't have looked very high.
            anomaly_score = max(anomaly_score, min(100.0, 70 + (max_abs_z - HARD_Z_GUARDRAIL) * 5))

        notable_params = {
            name: z for name, z in z_scores.items() if abs(z) >= Z_SCORE_NOTABLE
        }

        suspicious_param = None
        is_multi_parameter = False

        if is_anomaly:
            if len(notable_params) >= 2:
                is_multi_parameter = True
                # still report the single most extreme one for reference
                suspicious_param = max(notable_params, key=lambda n: abs(notable_params[n]))
            elif len(notable_params) == 1:
                suspicious_param = next(iter(notable_params))
            else:
                # Model flagged it as anomalous based on the joint
                # combination of features even though no single sensor
                # crosses the z-score threshold on its own.
                suspicious_param = max(z_scores, key=lambda n: abs(z_scores[n]))

        explanation = self._build_explanation(
            is_anomaly, is_multi_parameter, suspicious_param, z_scores, reading
        )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            suspicious_param=suspicious_param,
            param_deviations=z_scores,
            is_multi_parameter=is_multi_parameter,
            explanation=explanation,
            raw_values=reading,
        )

    # ------------------------------------------------------------------
    def _build_explanation(
        self,
        is_anomaly: bool,
        is_multi_parameter: bool,
        suspicious_param: Optional[str],
        z_scores: Dict[str, float],
        reading: Dict[str, float],
    ) -> str:
        if not is_anomaly:
            return "All sensor readings are consistent with the learned normal pattern."

        if is_multi_parameter:
            involved = [n for n, z in z_scores.items() if abs(z) >= Z_SCORE_NOTABLE]
            involved_str = ", ".join(i.replace("_", " ").title() for i in involved)
            return (
                f"Unusual multi-parameter condition detected involving "
                f"{involved_str} — requires verification. This could be a "
                f"genuine weather event or a multi-sensor fault; the system "
                f"cannot definitively distinguish between the two."
            )

        param_label = suspicious_param.replace("_", " ").title()
        z = z_scores[suspicious_param]
        direction = "above" if z > 0 else "below"
        return (
            f"{param_label} reading significantly deviates ({abs(z):.1f} standard "
            f"deviations {direction} the learned normal average) while other "
            f"sensors remain within their normal range."
        )
