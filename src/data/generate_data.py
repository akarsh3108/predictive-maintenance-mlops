"""
Synthetic sensor data generator for predictive maintenance.

Simulates industrial machines with realistic failure patterns:
- Temperature, vibration, pressure, rotation speed, torque, tool wear
- Failures become more likely as tool wear, temperature, and vibration rise
- Includes both gradual degradation and sudden failures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "pressure_bar",
]

TARGET_COLUMN = "failure"


def generate_sensor_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic sensor data with labeled failures.

    Failure rate is calibrated to ~3.5%, matching real industrial datasets
    (e.g. the AI4I 2020 predictive maintenance dataset).
    """
    rng = np.random.default_rng(seed)

    # Baseline operating conditions (normal distribution around nominal values)
    air_temp = rng.normal(300.0, 2.0, n_samples)  # Kelvin, ~27C
    process_temp = air_temp + rng.normal(10.0, 1.0, n_samples)
    rotational_speed = rng.normal(1540.0, 80.0, n_samples)  # rpm
    torque = rng.normal(40.0, 10.0, n_samples)  # Nm
    tool_wear = rng.uniform(0, 250, n_samples)  # minutes in operation
    vibration = rng.normal(2.5, 0.5, n_samples) + tool_wear / 500  # worsens with wear
    pressure = rng.normal(5.0, 0.3, n_samples)

    df = pd.DataFrame(
        {
            "air_temperature_k": air_temp,
            "process_temperature_k": process_temp,
            "rotational_speed_rpm": rotational_speed,
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
            "vibration_mm_s": vibration,
            "pressure_bar": pressure,
        }
    )

    # Failure logic: combination of risk factors raises failure probability.
    # We create a latent "stress" score, then convert it to a probability.
    heat_risk = np.clip((df["process_temperature_k"] - 308) / 5, 0, None)
    wear_risk = np.clip((df["tool_wear_min"] - 200) / 50, 0, None)
    vibration_risk = np.clip((df["vibration_mm_s"] - 3.0) / 1.0, 0, None)
    torque_risk = np.clip((df["torque_nm"] - 65) / 15, 0, None)
    speed_risk = np.clip((1380 - df["rotational_speed_rpm"]) / 100, 0, None)

    stress = heat_risk + wear_risk + vibration_risk + torque_risk + speed_risk
    # Logistic-style mapping to probability, tuned for ~3.5% base rate
    failure_prob = 1 / (1 + np.exp(-3.0 * (stress - 1.5)))
    # Add small random component (unpredictable sudden failures)
    failure_prob = np.clip(failure_prob + rng.normal(0, 0.005, n_samples), 0, 1)

    df[TARGET_COLUMN] = (rng.uniform(0, 1, n_samples) < failure_prob).astype(int)

    # Shuffle so failures aren't clustered at the end
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sensors.csv"))
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_sensor_data(n_samples=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    failure_rate = df[TARGET_COLUMN].mean()
    print(f"Generated {len(df):,} samples → {args.output}")
    print(f"Failure rate: {failure_rate:.2%} ({df[TARGET_COLUMN].sum():,} failures)")


if __name__ == "__main__":
    main()
