from __future__ import annotations

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
import redis
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

app = FastAPI(title="Fraud Detection API", version=APP_VERSION)

# Global state (initialized at startup)
model = None
feature_engineer = None
feature_columns: Optional[List[str]] = None
redis_client: Optional[redis.Redis] = None


# --------------------------
# Schemas (Pydantic v2-safe)
# --------------------------
class TransactionRequest(BaseModel):
    amount: float
    merchant_risk_score: float
    days_since_last_transaction: float
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: int = Field(ge=0, le=1)
    num_transactions_today: int = Field(ge=0)
    location_risk: float


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0)
    is_fraud: bool
    latency_ms: float
    model_version: str
    timestamp: str  # RFC3339 UTC


# --------------------------
# Helpers
# --------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_model_loaded():
    if model is None or feature_engineer is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

def _sorted_json_from_model(m: BaseModel) -> str:
    """Stable JSON for cache keys (sorted keys)."""
    d = m.model_dump()
    # hour_of_day and other ints must remain ints; ensure no numpy types sneak in
    return json.dumps(d, sort_keys=True, separators=(",", ":"))

def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(np.float32)
    return out


# --------------------------
# Startup
# --------------------------
@app.on_event("startup")
def load_artifacts():
    global model, feature_engineer, feature_columns, redis_client

    try:
        model_path = MODELS_DIR / "fraud_model.pkl"
        fe_path = MODELS_DIR / "feature_engineer.pkl"
        cols_path = MODELS_DIR / "feature_columns.pkl"

        if not model_path.exists() or not fe_path.exists() or not cols_path.exists():
            raise FileNotFoundError(
                f"Missing model artifacts in {MODELS_DIR}. "
                f"Expected: {model_path.name}, {fe_path.name}, {cols_path.name}"
            )

        model = joblib.load(model_path)
        feature_engineer = joblib.load(fe_path)
        feature_columns = joblib.load(cols_path)
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError("feature_columns is invalid or empty.")

        # Redis config (optional)
        redis_url = os.getenv("REDIS_URL")  # e.g., redis://localhost:6379/0
        if redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
        else:
            # fallback local
            try:
                redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
                redis_client.ping()
            except Exception:
                redis_client = None

        print(
            f"[startup] Model/FE loaded. features={len(feature_columns)} | "
            f"redis={'on' if redis_client else 'off'}"
        )
    except Exception as e:
        # Keep app up for /health, but indicate model not ready
        print(f"[startup] ERROR: {e}")
        model = None
        feature_engineer = None
        feature_columns = None
        redis_client = None


# --------------------------
# Endpoints
# --------------------------
@app.get("/")
def root():
    return {"app": app.title, "version": APP_VERSION, "model_version": MODEL_VERSION}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None and feature_engineer is not None and bool(feature_columns),
        "n_features": len(feature_columns) if feature_columns else 0,
        "cache_available": redis_client is not None,
        "timestamp": _utc_now_iso(),
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    _ensure_model_loaded()
    start = time.perf_counter()

    # Cache (stable key)
    response_json = None
    cache_key = None
    try:
        if redis_client:
            msg = _sorted_json_from_model(transaction)
            cache_key = "prediction:" + hashlib.md5(msg.encode("utf-8")).hexdigest()
            cached = redis_client.get(cache_key)
            if cached:
                # Validate JSON → model (ensures schema correctness)
                parsed = PredictionResponse.model_validate_json(cached)
                return parsed
    except Exception:
        # Cache errors should not affect prediction path
        cache_key = None

    # Prepare features
    df = pd.DataFrame([transaction.model_dump()])
    try:
        df_features = feature_engineer.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # Ensure all expected columns are present
    missing = [c for c in feature_columns if c not in df_features.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing expected feature columns: {missing}",
        )

    X = df_features[feature_columns]
    X = _to_float32(X)

    # Predict in threadpool (keeps event loop responsive)
    try:
        proba_arr = await run_in_threadpool(model.predict_proba, X)
        fraud_prob = float(proba_arr[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    resp = PredictionResponse(
        fraud_probability=fraud_prob,
        is_fraud=fraud_prob > 0.5,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
        timestamp=_utc_now_iso(),
    )

    # Write-through cache (best-effort)
    try:
        if redis_client and cache_key:
            redis_client.setex(cache_key, 300, resp.model_dump_json())
    except Exception:
        pass

    return resp


@app.get("/metrics")
def get_metrics():
    # Stub; wire to your monitoring/MLflow/prom exporter in prod
    try:
        return {
            "total_predictions": None,  # TODO: replace with a counter
            "avg_latency_ms": None,     # TODO: replace with actual histogram/summary
            "cache_hit_rate": None,     # TODO: compute from redis stats
            "fraud_detection_rate": 0.03,
            "timestamp": _utc_now_iso(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {e}")
