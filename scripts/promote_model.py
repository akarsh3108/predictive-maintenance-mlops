"""Promote an MLflow model version to a new stage.

Used by the Jenkins pipeline after manual approval to transition a
Staging model to Production. Also archives the previous Production
version so only one Production model is active at a time.
"""
from __future__ import annotations

import argparse
import os
import sys

import mlflow


MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance_model")


def promote(run_id: str, target_stage: str) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()

    # Find the model version that came from this run
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [v for v in all_versions if v.run_id == run_id]
    if not matching:
        sys.exit(f"No registered version found for run {run_id}")
    version = matching[0].version

    # Archive current Production if we're promoting something new to it
    if target_stage == "Production":
        current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in current_prod:
            if v.version != version:
                client.transition_model_version_stage(
                    name=MODEL_NAME, version=v.version, stage="Archived",
                )
                print(f"Archived previous Production v{v.version}")

    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage=target_stage,
    )
    print(f"Promoted {MODEL_NAME} v{version} → {target_stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=["Staging", "Production", "Archived"])
    args = parser.parse_args()
    promote(args.run_id, args.stage)


if __name__ == "__main__":
    main()
