"""Data drift detection.

Compares a recent window of production traffic against the training
distribution using the Kolmogorov-Smirnov test per feature. If too many
features drift significantly, we emit a signal to trigger retraining.

In production this would be run on a cron (e.g. every hour via a K8s
CronJob) against logged inference requests. The retraining trigger
could fire a Jenkins webhook or a Kubernetes Job.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

from src.data.generate_data import FEATURE_COLUMNS

# Per-feature p-value threshold. Low p-value = distributions differ.
DRIFT_P_THRESHOLD = 0.01
# If this fraction of features drift, we call the dataset drifted overall.
OVERALL_DRIFT_FRACTION = 0.3


@dataclass
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    p_value: float
    drifted: bool


@dataclass
class DriftReport:
    features: list[FeatureDriftResult]
    drifted_fraction: float
    overall_drift: bool
    n_reference: int
    n_current: int

    def to_dict(self) -> dict:
        return {
            "overall_drift": self.overall_drift,
            "drifted_fraction": self.drifted_fraction,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "features": [
                {
                    "feature": f.feature,
                    "ks_statistic": f.ks_statistic,
                    "p_value": f.p_value,
                    "drifted": f.drifted,
                }
                for f in self.features
            ],
        }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_threshold: float = DRIFT_P_THRESHOLD,
) -> DriftReport:
    """Run KS tests per feature and summarize.

    The KS test is distribution-free and picks up both mean shifts and
    shape changes, which makes it a reasonable default for sensor data.
    """
    results: list[FeatureDriftResult] = []
    for col in FEATURE_COLUMNS:
        if col not in reference.columns or col not in current.columns:
            continue
        ks_stat, p_value = stats.ks_2samp(reference[col], current[col])
        results.append(
            FeatureDriftResult(
                feature=col,
                ks_statistic=float(ks_stat),
                p_value=float(p_value),
                drifted=bool(p_value < p_threshold),
            )
        )

    drifted_count = sum(1 for r in results if r.drifted)
    drifted_fraction = drifted_count / len(results) if results else 0.0
    overall = drifted_fraction >= OVERALL_DRIFT_FRACTION

    return DriftReport(
        features=results,
        drifted_fraction=drifted_fraction,
        overall_drift=overall,
        n_reference=len(reference),
        n_current=len(current),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Training data CSV (reference distribution)",
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Recent production data CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the report",
    )
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)

    report = detect_drift(reference, current)
    report_dict = report.to_dict()

    print(json.dumps(report_dict, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report_dict, indent=2))

    # Exit code 2 signals drift → CI/CD can branch on this
    # (e.g. trigger a retraining Jenkins job).
    sys.exit(2 if report.overall_drift else 0)


if __name__ == "__main__":
    main()
