"""FastAPI inference service.

Loads the latest Production model from the MLflow registry on startup.
Exposes:
  - POST /predict      — single or batch predictions
  - GET  /health       — Kubernetes liveness probe
  - GET  /ready        — Kubernetes readiness probe (returns 503 until model loaded)
  - GET  /metrics      — Prometheus scrape endpoint
  - POST /reload       — admin endpoint to reload model without restarting pod
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.data.generate_data import FEATURE_COLUMNS

MODEL_NAME = os.getenv("MODEL_NAME", "predictive_maintenance_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
DEFAULT_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

logger = logging.getLogger("pdm.serving")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)


# ---------- Prometheus metrics ----------
# These are scraped by the Kubernetes Prometheus operator and visualized in Grafana.
PREDICTIONS_TOTAL = Counter(
    "pdm_predictions_total",
    "Total predictions made",
    ["outcome"],
)
PREDICTION_LATENCY = Histogram(
    "pdm_prediction_latency_seconds",
    "Inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTION_SCORE = Histogram(
    "pdm_prediction_score",
    "Distribution of predicted failure probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOAD_ERRORS = Counter("pdm_model_load_errors_total", "Failures loading model")


# ---------- Pydantic schemas ----------
class SensorReading(BaseModel):
    """One sample of sensor telemetry. Field names must match FEATURE_COLUMNS."""

    air_temperature_k: float = Field(..., ge=250, le=400)
    process_temperature_k: float = Field(..., ge=250, le=400)
    rotational_speed_rpm: float = Field(..., ge=0, le=5000)
    torque_nm: float = Field(..., ge=0, le=200)
    tool_wear_min: float = Field(..., ge=0, le=1000)
    vibration_mm_s: float = Field(..., ge=0, le=50)
    pressure_bar: float = Field(..., ge=0, le=50)


class PredictRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)
    threshold: float | None = Field(None, ge=0, le=1)


class Prediction(BaseModel):
    failure_probability: float
    will_fail: bool
    threshold_used: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    latency_ms: float


# ---------- Model holder ----------
class ModelBundle:
    """Holds the loaded PyTorch model, preprocessing scaler, and version metadata."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.scaler: Any | None = None
        self.version: str = "unloaded"
        self.loaded_at: float = 0.0

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        """Load the latest model from MLflow registry.

        Pulls both the model and the preprocessing scaler artifact that
        was logged alongside it during training.
        """
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if not versions:
                raise RuntimeError(
                    f"No model found in stage '{MODEL_STAGE}'. "
                    f"Train a model first and promote it."
                )
            mv = versions[0]

            logger.info(
                "Loading model %s v%s from stage %s (run_id=%s)",
                MODEL_NAME,
                mv.version,
                MODEL_STAGE,
                mv.run_id,
            )

            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()

            # Download the scaler artifact from the run that produced this model
            scaler_local = client.download_artifacts(
                run_id=mv.run_id,
                path="preprocessing/scaler.pkl",
            )
            self.scaler = joblib.load(scaler_local)

            self.version = mv.version
            self.loaded_at = time.time()
            logger.info("Model loaded successfully (version=%s)", self.version)

        except Exception:
            MODEL_LOAD_ERRORS.inc()
            logger.exception("Failed to load model")
            raise


bundle = ModelBundle()


# ---------- FastAPI app lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup. If loading fails, the pod won't pass readiness."""
    try:
        bundle.load()
    except Exception:
        # Don't crash the pod — readiness probe will return 503 so K8s
        # won't route traffic, but the /reload endpoint stays available.
        logger.error("Startup model load failed; /ready will return 503")
    yield


app = FastAPI(
    title="Predictive Maintenance Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Endpoints ----------
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — returns OK as long as the process is alive."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness probe — returns 503 until the model is loaded."""
    if not bundle.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": bundle.version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not bundle.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
    start = time.perf_counter()

    # Build feature matrix in the same column order as training
    X = np.array(
        [[getattr(r, col) for col in FEATURE_COLUMNS] for r in req.readings],
        dtype=np.float32,
    )
    X_scaled = bundle.scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        logits = bundle.model(torch.from_numpy(X_scaled))
        probs = torch.sigmoid(logits).cpu().numpy()

    predictions = []
    for p in probs:
        p_float = float(p)
        will_fail = p_float >= threshold
        predictions.append(
            Prediction(
                failure_probability=p_float,
                will_fail=will_fail,
                threshold_used=threshold,
            )
        )
        PREDICTIONS_TOTAL.labels(outcome="failure" if will_fail else "healthy").inc()
        PREDICTION_SCORE.observe(p_float)

    latency = time.perf_counter() - start
    PREDICTION_LATENCY.observe(latency)

    return PredictResponse(
        predictions=predictions,
        model_version=bundle.version,
        latency_ms=latency * 1000,
    )


@app.post("/reload")
def reload_model(request: Request) -> dict[str, str]:
    """Reload the model without restarting the pod.

    Called by the retraining pipeline after promoting a new Production version.
    Protected by a simple shared-secret header in production deployments.
    """
    expected_token = os.getenv("RELOAD_TOKEN")
    if expected_token:
        provided = request.headers.get("X-Reload-Token")
        if provided != expected_token:
            raise HTTPException(status_code=401, detail="Invalid reload token")

    try:
        bundle.load()
        return {"status": "reloaded", "model_version": bundle.version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # one worker per pod; scale horizontally via K8s HPA
    )
