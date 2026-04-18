"""Data loading, schema validation, and preprocessing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.generate_data import FEATURE_COLUMNS, TARGET_COLUMN


class SchemaError(ValueError):
    """Raised when input data does not match expected schema."""


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    data_hash: str


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Validate DataFrame has required columns and no nulls.

    Raises SchemaError with a helpful message if validation fails.
    Used both in training (require_target=True) and serving (require_target=False).
    """
    required = set(FEATURE_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)

    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"Missing required columns: {sorted(missing)}")

    null_counts = df[list(required)].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        raise SchemaError(f"Null values found in columns: {null_cols.to_dict()}")

    # Sanity checks on sensor ranges — catches upstream data pipeline bugs
    if (df["rotational_speed_rpm"] < 0).any():
        raise SchemaError("Negative rotational speed detected")
    if (df["tool_wear_min"] < 0).any():
        raise SchemaError("Negative tool wear detected")


def compute_data_hash(df: pd.DataFrame) -> str:
    """Produce a stable hash of the dataframe for reproducibility tracking.

    Logged to MLflow so we can tell whether two runs used identical data.
    """
    # Hash the raw bytes of a canonical CSV representation
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()[:16]


def load_and_split(
    csv_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Load CSV, validate, split into train/val/test, and fit a scaler.

    The scaler is fit on TRAIN ONLY to prevent leakage into val/test.
    """
    df = pd.read_csv(csv_path)
    validate_schema(df, require_target=True)
    data_hash = compute_data_hash(df)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(np.float32)

    # Stratified split — preserves failure rate across splits (critical for imbalanced data)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )
    # Adjust val_size to be a fraction of the remaining data, not the original
    val_fraction_of_remaining = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_remaining,
        stratify=y_trainval,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return DataSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        scaler=scaler,
        feature_names=list(FEATURE_COLUMNS),
        data_hash=data_hash,
    )
