"""
sensor_simulator.py
=====================
A virtual Automatic Weather Station (AWS). Since the team does not have
physical sensor hardware for this first prototype, this module generates
realistic, continuously-updating sensor readings.

It supports two modes:
    1. NORMAL mode - values vary naturally around realistic baselines.
    2. ANOMALY INJECTION mode - a specific parameter is deliberately
       pushed to an abnormal value for demonstration purposes, while the
       others keep behaving normally (this is what lets the ML model
       show that it can isolate the ONE odd-one-out sensor).

This module intentionally knows nothing about the ML model - it only
produces data. Detection logic lives entirely in ml/anomaly_detector.py.
"""

import math
import random
from datetime import datetime
from typing import Dict, Optional


class SensorSimulator:
    # Reference stats (approximate mean/std as actually produced by the
    # generators below) - used only for display units and for building
    # "physically odd" anomaly spike magnitudes. IMPORTANT: the NORMAL
    # generation formulas below are deliberately written to match the
    # distributions used in ml/train_model.py's generate_normal_weather_data,
    # so that what the simulator calls "normal" is the same "normal" the
    # ML model was trained on.
    BASELINES = {
        "temperature": {"mean": 28.0, "std": 4.3, "unit": "°C"},
        "humidity": {"mean": 75.0, "std": 4.5, "unit": "%"},
        "pressure": {"mean": 1010.0, "std": 1.4, "unit": "hPa"},
        "wind_speed": {"mean": 12.0, "std": 8.3, "unit": "km/h"},
        "rainfall": {"mean": 0.25, "std": 1.4, "unit": "mm"},
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._t = 0  # internal "tick" counter, drives a gentle day cycle
        self._pressure_drift = 0.0  # slow random-walk state for pressure
        self._active_anomaly: Optional[str] = None
        # how many more readings the injected anomaly should persist for
        self._anomaly_ticks_remaining = 0

    # ------------------------------------------------------------------
    def inject_anomaly(self, parameter: str, duration_readings: int = 12):
        """Activates an anomaly on the given parameter for the next
        `duration_readings` calls to generate_reading(). Other parameters
        keep behaving normally, so the ML model has to pick out the one
        odd sensor from the group."""
        if parameter not in self.BASELINES:
            raise ValueError(f"Unknown parameter: {parameter}")
        self._active_anomaly = parameter
        self._anomaly_ticks_remaining = duration_readings

    def reset(self):
        """Clears any active anomaly injection and returns to normal mode."""
        self._active_anomaly = None
        self._anomaly_ticks_remaining = 0

    @property
    def active_anomaly(self) -> Optional[str]:
        return self._active_anomaly

    # ------------------------------------------------------------------
    def _normal_reading(self) -> Dict[str, float]:
        """
        Generates one full set of NORMAL sensor values. These formulas are
        intentionally the same shape as ml/train_model.py's
        generate_normal_weather_data(), so live "normal" simulation stays
        consistent with what the model was actually trained to recognise
        as normal (temperature daily cycle, inverse humidity relationship,
        slow-drifting pressure, skewed wind distribution, bursty rainfall).
        """
        daily_cycle = math.sin(2 * math.pi * (self._t % 288) / 288)

        temperature = 28 + 6 * daily_cycle + self._rng.gauss(0, 0.8)

        humidity = 75 - 4 * daily_cycle + self._rng.gauss(0, 3)
        humidity = max(30.0, min(95.0, humidity))

        # slow bounded random-walk drift, mirrors the cumulative drift
        # used when generating the training dataset
        self._pressure_drift += self._rng.gauss(0, 0.05)
        self._pressure_drift = max(-8.0, min(8.0, self._pressure_drift))
        pressure = 1010 + self._pressure_drift + self._rng.gauss(0, 0.3)

        # gamma(shape=2, scale=6) via sum of exponential draws (simple,
        # dependency-free approximation of numpy's gamma sampling)
        wind_speed = sum(self._rng.expovariate(1 / 6.0) for _ in range(2))
        wind_speed = max(0.0, min(45.0, wind_speed))

        if self._rng.random() < 0.06:
            rainfall = self._rng.expovariate(1 / 4.0)
        else:
            rainfall = 0.0
        rainfall = max(0.0, min(40.0, rainfall))

        return {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
        }

    def _anomalous_value(self, param: str) -> float:
        """Builds a physically-plausible-but-clearly-abnormal value for
        the given parameter, well outside its normal operating band."""
        cfg = self.BASELINES[param]
        # Wind speed and rainfall are naturally floored at 0 (calm/dry
        # conditions are completely normal, not anomalous), so a
        # meaningful demo spike for these two can only go UP. Temperature,
        # humidity and pressure can plausibly spike in either direction.
        if param in ("wind_speed", "rainfall"):
            spike_direction = 1
        else:
            spike_direction = self._rng.choice([1, -1])
        # push the value 6-10 standard deviations away from the mean
        magnitude = self._rng.uniform(6, 10) * cfg["std"]
        value = cfg["mean"] + spike_direction * magnitude

        # keep it within "sensor could plausibly report this" bounds
        # rather than absurd/impossible numbers
        if param == "temperature":
            value = max(-10, min(70, value))
        elif param == "humidity":
            value = max(0, min(100, value))
        elif param == "pressure":
            value = max(950, min(1070, value))
        elif param == "wind_speed":
            value = max(0, min(150, value))
        elif param == "rainfall":
            value = max(0, min(120, value))
        return value

    # ------------------------------------------------------------------
    def generate_reading(self) -> Dict:
        """Produces ONE new reading (dict of 5 sensor values + metadata)."""
        self._t += 1

        normal_values = self._normal_reading()
        reading = {p: round(normal_values[p], 2) for p in self.BASELINES}

        if self._active_anomaly and self._anomaly_ticks_remaining > 0:
            param = self._active_anomaly
            reading[param] = round(self._anomalous_value(param), 2)
            self._anomaly_ticks_remaining -= 1
            if self._anomaly_ticks_remaining == 0:
                self._active_anomaly = None

        reading["timestamp"] = datetime.now()
        return reading

    @classmethod
    def units(cls) -> Dict[str, str]:
        return {p: cfg["unit"] for p, cfg in cls.BASELINES.items()}
