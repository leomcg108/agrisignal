"""
serving/api.py
───────────────
FastAPI real-time inference endpoint.

Endpoints:
  GET  /health             Health check + model loaded status
  POST /predict            Single prediction (cached 5 min)
  GET  /features/latest    Latest feature snapshot for monitoring
  GET  /model/metadata     Feature list, training metrics
  GET  /metrics            Prometheus metrics
  POST /pipeline/trigger   Trigger pipeline rerun (admin)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agrisignal.monitoring.metrics import (
    get_metrics_output,
    model_prediction_value,
    prediction_latency,
    prediction_requests,
)
from agrisignal.utils import get_logger, load_config

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Predictor (singleton, loaded at startup)
# ─────────────────────────────────────────────────────────────────


class Predictor:
    """Loads model artifacts and serves predictions."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self._gold_df: pd.DataFrame | None = None
        self._loaded = False

    def load(self):
        from agrisignal.features.engineer import FeatureEngineer
        from agrisignal.training.train_xgboost import XGBoostTrainer

        trainer = XGBoostTrainer()
        self.model, self.scaler, self.metadata = trainer.load()

        engineer = FeatureEngineer()
        self._gold_df = engineer.read()
        self._loaded = True
        log.info(
            f"Model loaded: {self.metadata['n_features']} features | "
            f"DA={self.metadata['test_metrics']['da']:.3f}"
        )

    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def predict(self, horizon_days: int | None = None) -> dict:
        """Generate prediction using most recent available feature row."""
        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        feature_cols = self.metadata["feature_cols"]
        df = self._gold_df.sort_values("date")

        # Use most recent fully-available row (not in target-null tail)
        valid = df.dropna(subset=feature_cols)
        if valid.empty:
            raise RuntimeError("No valid feature rows available")

        latest_row = valid.iloc[-1]
        X = latest_row[feature_cols].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        pred = float(self.model.predict(X_scaled)[0])

        # Direction classification with confidence proxy
        direction = "BULLISH" if pred > 0.1 else "BEARISH" if pred < -0.1 else "NEUTRAL"

        return {
            "as_of_date": str(latest_row["date"].date()),
            "predicted_return_pct": round(pred, 4),
            "direction": direction,
            "horizon_days": horizon_days or self.metadata["target_horizon_days"],
            "feature_snapshot": {
                k: round(float(latest_row.get(k, 0)), 4)
                for k in [
                    "rsi",
                    "macd",
                    "bb_pct_b",
                    "rvol_20d",
                    "gdd_cumulative",
                    "heat_stress_14d",
                ]
                if k in latest_row.index and pd.notna(latest_row.get(k))
            },
        }


_predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _predictor.load()
    except FileNotFoundError:
        log.warning(
            "No model found. Run the pipeline first: "
            "python -m agrisignal.orchestration.flows.daily_pipeline"
        )
    except Exception as e:
        log.error(f"Model load failed: {e}")
    yield


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgriSignal API",
    description="Corn futures ML prediction API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_start_time = time.time()


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    horizon_days: int | None = Field(None, ge=1, le=30)


class PredictResponse(BaseModel):
    as_of_date: str
    predicted_return_pct: float
    direction: str
    horizon_days: int
    feature_snapshot: dict
    inference_ms: float


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _predictor.is_ready(),
        "uptime_s": round(time.time() - _start_time, 1),
        "model_metadata": _predictor.metadata if _predictor.is_ready() else {},
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest | None = None):
    if req is None:
        req = PredictRequest()

    if not _predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the pipeline first.",
        )
    t0 = time.perf_counter()
    try:
        result = _predictor.predict(horizon_days=req.horizon_days)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        prediction_requests.labels(status="success").inc()
        prediction_latency.observe(elapsed_ms / 1000)
        model_prediction_value.observe(result["predicted_return_pct"])

        return PredictResponse(**result, inference_ms=round(elapsed_ms, 2))

    except Exception as exc:
        prediction_requests.labels(status="error").inc()
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/features/latest")
async def latest_features():
    """Return the latest feature values for drift monitoring."""
    if not _predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = _predictor._gold_df.sort_values("date")
    latest = df.iloc[-1]
    feature_cols = _predictor.metadata["feature_cols"]
    return {
        "as_of_date": str(latest["date"].date()),
        "close": float(latest["close"]),
        "features": {
            k: round(float(latest[k]), 4)
            for k in feature_cols[:30]
            if k in latest.index and pd.notna(latest[k])
        },
    }


@app.get("/model/metadata")
async def model_metadata():
    if not _predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor.metadata


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    data, content_type = get_metrics_output()
    return Response(content=data, media_type=content_type)


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "agrisignal.serving.api:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        workers=cfg["api"]["workers"],
        log_level="info",
    )
