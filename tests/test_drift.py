"""Tests for the drift detection module."""
from __future__ import annotations

import pandas as pd

from src.data.generate_data import generate_sensor_data
from src.monitoring.drift import detect_drift


def test_no_drift_on_identical_data():
    df = generate_sensor_data(n_samples=2000, seed=42)
    report = detect_drift(df, df.copy())
    assert not report.overall_drift
    assert report.drifted_fraction == 0.0


def test_no_drift_on_same_distribution_different_sample():
    ref = generate_sensor_data(n_samples=2000, seed=42)
    cur = generate_sensor_data(n_samples=2000, seed=43)  # different seed, same process
    report = detect_drift(ref, cur)
    # With same generating process, most features shouldn't drift
    assert report.drifted_fraction < 0.3


def test_detects_drift_on_shifted_data():
    ref = generate_sensor_data(n_samples=2000, seed=42)
    cur = ref.copy()
    # Simulate sensor calibration drift: temperatures rise 10K, vibration doubles
    cur["air_temperature_k"] += 10
    cur["process_temperature_k"] += 10
    cur["vibration_mm_s"] *= 2
    cur["torque_nm"] += 15

    report = detect_drift(ref, cur)
    assert report.overall_drift, f"Failed to detect clear drift: {report.to_dict()}"
    drifted_features = [f.feature for f in report.features if f.drifted]
    assert "air_temperature_k" in drifted_features
    assert "vibration_mm_s" in drifted_features


def test_drift_report_structure():
    df = generate_sensor_data(n_samples=500, seed=42)
    report = detect_drift(df, df.copy())
    d = report.to_dict()
    assert "overall_drift" in d
    assert "features" in d
    assert len(d["features"]) > 0
    for f in d["features"]:
        assert set(f.keys()) == {"feature", "ks_statistic", "p_value", "drifted"}
