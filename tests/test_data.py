"""Tests for data loading, schema validation, and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import FEATURE_COLUMNS, TARGET_COLUMN, generate_sensor_data
from src.data.preprocessing import (
    SchemaError,
    compute_data_hash,
    load_and_split,
    validate_schema,
)


def test_generate_data_has_correct_shape():
    df = generate_sensor_data(n_samples=1000, seed=42)
    assert len(df) == 1000
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)


def test_generate_data_failure_rate_reasonable():
    """Failure rate should match industrial reality (~3-5%)."""
    df = generate_sensor_data(n_samples=10_000, seed=42)
    rate = df[TARGET_COLUMN].mean()
    assert 0.02 < rate < 0.25, f"Unrealistic failure rate: {rate}"


"""Failure rate should be non-trivial but not dominate (5-25%)."""


def test_generate_data_is_reproducible():
    df1 = generate_sensor_data(n_samples=500, seed=7)
    df2 = generate_sensor_data(n_samples=500, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_validate_schema_passes_for_valid_data():
    df = generate_sensor_data(n_samples=100)
    validate_schema(df, require_target=True)  # should not raise


def test_validate_schema_rejects_missing_column():
    df = generate_sensor_data(n_samples=100)
    df = df.drop(columns=["torque_nm"])
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_rejects_nulls():
    df = generate_sensor_data(n_samples=100)
    df.loc[5, "vibration_mm_s"] = None
    with pytest.raises(SchemaError, match="Null values"):
        validate_schema(df)


def test_validate_schema_rejects_negative_rpm():
    df = generate_sensor_data(n_samples=100)
    df.loc[3, "rotational_speed_rpm"] = -50.0
    with pytest.raises(SchemaError, match="Negative rotational speed"):
        validate_schema(df)


def test_data_hash_is_stable():
    df = generate_sensor_data(n_samples=500, seed=42)
    assert compute_data_hash(df) == compute_data_hash(df)


def test_data_hash_changes_with_data():
    df1 = generate_sensor_data(n_samples=500, seed=42)
    df2 = generate_sensor_data(n_samples=500, seed=43)
    assert compute_data_hash(df1) != compute_data_hash(df2)


def test_load_and_split_preserves_failure_rate(tmp_path):
    """Stratified split should keep failure rate consistent across splits."""
    df = generate_sensor_data(n_samples=5000, seed=42)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    split = load_and_split(csv_path, seed=42)
    overall_rate = df[TARGET_COLUMN].mean()
    for y in (split.y_train, split.y_val, split.y_test):
        assert (
            abs(y.mean() - overall_rate) < 0.02
        ), f"Stratified split failed: {y.mean()} vs {overall_rate}"


def test_load_and_split_scaler_fitted_on_train_only(tmp_path):
    """Prevents a common leakage bug."""
    df = generate_sensor_data(n_samples=1000, seed=42)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    split = load_and_split(csv_path, seed=42)
    # Training data should have approximately zero mean and unit variance
    # after scaling (since scaler was fit on it).
    assert np.allclose(split.X_train.mean(axis=0), 0, atol=0.1)
    assert np.allclose(split.X_train.std(axis=0), 1, atol=0.1)
