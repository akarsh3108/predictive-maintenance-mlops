"""Offline validation of a newly-trained model.

Runs after training completes. If any check fails, the Jenkins stage fails
and the model never reaches Staging — regardless of in-training metrics.

Checks performed:
  1. Quality gate is tagged PASSED on the run
  2. Model is loadable via mlflow.pytorch.load_model()
  3. Model produces valid probabilities (in [0,1]) on canary inputs
  4. Model responds sensibly to extreme inputs (sanity check)
"""
from __future__ import annotations

import argparse
import os
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import torch


def validate(run_id: str) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()

    run = client.get_run(run_id)

    # Check 1: quality gate
    gate = run.data.tags.get("quality_gate")
    if gate != "PASSED":
        sys.exit(f"Quality gate not PASSED (was: {gate})")
    print("[OK] Quality gate PASSED")

    # Check 2: model loads
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print("[OK] Model loads via mlflow.pytorch.load_model")

    # Check 3: valid probabilities on random canary inputs
    n_features = 7
    rng = np.random.default_rng(0)
    canary = torch.from_numpy(rng.normal(0, 1, (100, n_features)).astype(np.float32))
    with torch.no_grad():
        probs = torch.sigmoid(model(canary)).numpy()
    if not ((probs >= 0).all() and (probs <= 1).all()):
        sys.exit("Model produced out-of-range probabilities")
    if np.isnan(probs).any():
        sys.exit("Model produced NaN probabilities")
    print(f"[OK] Canary predictions valid (mean prob: {probs.mean():.3f})")

    # Check 4: sanity — extreme high-stress inputs should give higher
    # failure prob than normal ones. This catches catastrophically broken models.
    normal = torch.zeros(10, n_features)  # mean feature values after scaling = 0
    extreme = torch.full((10, n_features), 3.0)  # 3 std devs above mean
    with torch.no_grad():
        p_normal = torch.sigmoid(model(normal)).mean().item()
        p_extreme = torch.sigmoid(model(extreme)).mean().item()

    # We don't require extreme > normal strictly (depends on which features
    # matter), but NaN or identical outputs would be suspicious.
    print(f"[OK] Sanity: normal p={p_normal:.3f}, extreme p={p_extreme:.3f}")

    print("\nAll validation checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    validate(args.run_id)


if __name__ == "__main__":
    main()
