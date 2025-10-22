import os
import importlib
from pathlib import Path

import pytest

# ---- Helpers to build a tiny model bundle for tests ----
@pytest.fixture(scope="session")
def tmp_models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models_bundle")
    return d

@pytest.fixture(scope="session")
def build_artifacts(tmp_models_dir):
    """
    Train a very small model and write the three expected artifacts:
    fraud_model.pkl, feature_engineer.pkl, feature_columns.pkl
    """
    # Import here to avoid importing app prematurely
    from src.data.generator import FraudDataGenerator
    from src.models.fraud_model import FraudModel
    import joblib

    # Small dataset for speed
    df = FraudDataGenerator(n_samples=3000, target_rate=0.03, seed=123).generate()

    # Train a quick baseline (no xgboost required in CI)
    model = FraudModel(model_type="random_forest", random_state=7)
    _ = model.train(df)  # saves to ./models by default

    # Move or re-save artifacts into tmp_models_dir expected by API
    src_dir = Path("models")
    want = ["fraud_model.pkl", "feature_engineer.pkl", "feature_columns.pkl"]
    for name in want:
        obj = joblib.load(src_dir / name)
        joblib.dump(obj, tmp_models_dir / name)

    return tmp_models_dir

@pytest.fixture
def app_with_artifacts(monkeypatch, build_artifacts):
    """
    Set MODELS_DIR env var BEFORE importing the FastAPI app module,
    so its startup uses our temp artifacts.
    """
    # 1) Point API to our temp models dir
    monkeypatch.setenv("MODELS_DIR", str(build_artifacts))
    # Optional: ensure Redis is not required in CI
    monkeypatch.delenv("REDIS_URL", raising=False)

    # 2) Import (or reload) the app module AFTER env is set
    #    so that module-level constants pick up the new MODELS_DIR.
    from src.api import app as app_module
    importlib.reload(app_module)  # pick up new env
    return app_module.app  # FastAPI instance


# ---- Tests ----

def test_health_check(app_with_artifacts):
    from fastapi.testclient import TestClient
    with TestClient(app_with_artifacts) as client:
        r = client.get("/health")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "healthy"
        assert payload["model_loaded"] is True
        assert payload["n_features"] > 0

def test_prediction(app_with_artifacts):
    from fastapi.testclient import TestClient
    with TestClient(app_with_artifacts) as client:
        transaction = {
            "amount": 150.0,
            "merchant_risk_score": 0.3,
            "days_since_last_transaction": 1.0,
            "hour_of_day": 14,
            "is_weekend": 0,
            "num_transactions_today": 3,
            "location_risk": 0.2,
        }
        r = client.post("/predict", json=transaction)
        assert r.status_code == 200, r.text
        result = r.json()
        assert "fraud_probability" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0
        # Optional extra checks
        assert isinstance(result["is_fraud"], bool)
        assert "latency_ms" in result and result["latency_ms"] >= 0
        assert result["model_version"]  # non-empty
