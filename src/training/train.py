"""Training pipeline with MLflow tracking and automated model registration.

Key MLOps behaviors:
  1. Every run logs code version (git SHA), hyperparameters, data hash, metrics.
  2. On success, the model is logged as an MLflow artifact.
  3. If the new model beats the registered Production model on test AUC,
     it's promoted to 'Staging' (human approval still required for Production).
  4. Otherwise it's logged but left in 'None' stage.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import load_and_split
from src.models.classifier import FailurePredictor


MODEL_NAME = "predictive_maintenance_model"


@dataclass
class TrainConfig:
    data_path: str
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.2
    # Quality gate — model won't be registered if test AUC falls below this
    min_test_auc: float = 0.85
    seed: int = 42
    mlflow_experiment: str = "predictive_maintenance"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # hidden_dims comes in as list from YAML; convert to tuple
        if "hidden_dims" in data:
            data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**data)


def get_git_sha() -> str:
    """Capture code version for reproducibility. Returns 'unknown' if not a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """Compute positive class weight for BCEWithLogitsLoss.

    With ~3.5% failure rate, we weight positives ~27x to prevent the model
    from collapsing to 'always predict no failure'.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Compute AUC, PR-AUC, and F1 at the optimal threshold."""
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    probs = torch.sigmoid(model(X_t)).cpu().numpy()

    auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)

    # Find threshold that maximizes F1 on this set
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last point has no threshold
    best_threshold = float(thresholds[best_idx])
    best_f1 = f1_score(y, (probs >= best_threshold).astype(int))

    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": float(best_f1),
        "optimal_threshold": best_threshold,
    }


def maybe_promote_to_staging(run_id: str, test_auc: float) -> str:
    """Compare to current Production model; promote if better.

    Returns the stage the new model was placed in.
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Register the new version
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    new_version = mv.version

    # Look up current Production model's test AUC
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.RestException:
        prod_versions = []

    if not prod_versions:
        # No Production yet — promote this one to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME, version=new_version, stage="Staging",
        )
        print(f"[registry] v{new_version} → Staging (no existing Production)")
        return "Staging"

    prod_run = client.get_run(prod_versions[0].run_id)
    prod_auc = prod_run.data.metrics.get("test_auc", 0.0)

    if test_auc > prod_auc:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=new_version, stage="Staging",
        )
        print(
            f"[registry] v{new_version} → Staging "
            f"(test_auc {test_auc:.4f} > Production {prod_auc:.4f})"
        )
        return "Staging"
    else:
        print(
            f"[registry] v{new_version} stays in 'None' "
            f"(test_auc {test_auc:.4f} ≤ Production {prod_auc:.4f})"
        )
        return "None"


def train(config: TrainConfig) -> dict[str, float]:
    """Main training entry point. Returns final test metrics."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    split = load_and_split(Path(config.data_path), seed=config.seed)
    print(
        f"Data loaded (hash={split.data_hash}): "
        f"train={len(split.y_train)}, val={len(split.y_val)}, test={len(split.y_test)}"
    )

    train_ds = TensorDataset(
        torch.from_numpy(split.X_train), torch.from_numpy(split.y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    # Model, loss (with class balancing), optimizer
    model = FailurePredictor(
        n_features=split.X_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    pos_weight = compute_pos_weight(split.y_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    # MLflow experiment tracking
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run() as run:
        # Log config, code version, data provenance
        mlflow.log_params(asdict(config))
        mlflow.log_param("git_sha", get_git_sha())
        mlflow.log_param("data_hash", split.data_hash)
        mlflow.log_param("n_train", len(split.y_train))
        mlflow.log_param("n_val", len(split.y_val))
        mlflow.log_param("n_test", len(split.y_test))
        mlflow.log_param("train_failure_rate", float(split.y_train.mean()))

        best_val_auc = 0.0
        for epoch in range(1, config.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_metrics = evaluate(model, split.X_val, split.y_val, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]

            if epoch % 5 == 0 or epoch == config.epochs:
                print(
                    f"  epoch {epoch:3d} | loss {train_loss:.4f} | "
                    f"val_auc {val_metrics['auc']:.4f} | val_f1 {val_metrics['f1']:.4f}"
                )

        # Final evaluation on held-out test set
        test_metrics = evaluate(model, split.X_test, split.y_test, device)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        print(f"\nTest metrics: {test_metrics}")

        # Quality gate — reject model if below threshold
        if test_metrics["auc"] < config.min_test_auc:
            mlflow.set_tag("quality_gate", "FAILED")
            raise RuntimeError(
                f"Model failed quality gate: test_auc={test_metrics['auc']:.4f} "
                f"< min_test_auc={config.min_test_auc}"
            )
        mlflow.set_tag("quality_gate", "PASSED")

        # Log the model along with scaler (as artifact) so serving can preprocess
        import joblib, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.pkl"
            joblib.dump(split.scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=None,  # we register manually below
        )

        # Promote if it beats Production
        stage = maybe_promote_to_staging(run.info.run_id, test_metrics["auc"])
        mlflow.set_tag("stage_after_run", stage)

        return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    # Allow tracking URI override from environment (used by Jenkins/K8s)
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    train(config)


if __name__ == "__main__":
    main()
