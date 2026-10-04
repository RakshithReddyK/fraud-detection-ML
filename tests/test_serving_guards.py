import json

import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.data.generator import FraudDataGenerator
from src.models.trainer import FraudModel

TX = {
    "amount": 150.0,
    "merchant_risk_score": 0.3,
    "days_since_last_transaction": 1.0,
    "hour_of_day": 14,
    "is_weekend": 0,
    "num_transactions_today": 3,
    "location_risk": 0.2,
}


def test_feature_statistics_fit_training_rows_only(tmp_path):
    df = FraudDataGenerator(n_samples=2000, seed=71).generate()
    train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_fraud)
    model = FraudModel(
        "random_forest",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        track_mlflow=False,
    )
    model.train(df)
    assert model.feature_engineer.feature_stats["amount_mean"] == pytest.approx(train.amount.mean())
    assert model.feature_engineer.feature_stats["amount_mean"] != pytest.approx(df.amount.mean())
    metrics = json.loads((tmp_path / "reports/evaluation.json").read_text())
    assert metrics["baseline_average_precision"] == pytest.approx(metrics["test_prevalence"])
    assert metrics["feature_statistics_fit_on"] == "training_rows_only"


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount", -1),
        ("merchant_risk_score", 1.1),
        ("location_risk", -0.1),
        ("days_since_last_transaction", -2),
    ],
)
def test_invalid_transactions(app_with_artifacts, field, value):
    with TestClient(app_with_artifacts) as c:
        assert c.post("/predict", json={**TX, field: value}).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_returns_validation_error(app_with_artifacts, value):
    with TestClient(app_with_artifacts) as c:
        response = c.post(
            "/predict",
            content=json.dumps({**TX, "amount": value}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "amount"]
        assert "input" not in response.json()["detail"][0]


def test_unavailable_redis_keeps_model_ready(app_with_artifacts, monkeypatch):
    import redis

    from src.api import main

    class OfflineCache:
        def ping(self):
            raise redis.ConnectionError("unavailable")

    monkeypatch.setenv("REDIS_URL", "redis://localhost:1")
    monkeypatch.setattr(main.redis, "from_url", lambda *a, **kw: OfflineCache())
    with TestClient(app_with_artifacts) as c:
        assert c.get("/ready").status_code == 200
        assert c.post("/predict", json=TX).status_code == 200
        assert c.get("/health").json()["cache_available"] is False


def test_missing_artifact_is_not_ready(app_with_artifacts, monkeypatch, tmp_path):
    from src.api import main

    monkeypatch.setattr(main, "MODELS_DIR", tmp_path)
    with TestClient(app_with_artifacts) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=TX).status_code == 503


def test_cache_hit_has_fresh_timing_and_bundle_key(app_with_artifacts):
    from src.api import main

    class Cache:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def setex(self, key, ttl, value):
            self.values[key] = value

    with TestClient(app_with_artifacts) as c:
        cache = Cache()
        main.redis_client = cache
        a = c.post("/predict", json=TX).json()
        b = c.post("/predict", json=TX).json()
        assert b["timestamp"] != a["timestamp"]
        assert b["fraud_probability"] == a["fraud_probability"]
        assert all(main.MODEL_VERSION in key for key in cache.values)
        assert main.serving_metrics.snapshot()["cache_hit_rate"] > 0
