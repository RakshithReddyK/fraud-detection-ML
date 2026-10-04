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


def test_metrics(app_with_artifacts):
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
        # /metrics should reflect real counters, not hardcoded placeholders
        before = client.get("/metrics").json()
        assert before["total_predictions"] >= 0

        client.post("/predict", json=transaction)

        after = client.get("/metrics").json()
        assert after["total_predictions"] == before["total_predictions"] + 1
        assert after["avg_latency_ms"] >= 0
        assert 0.0 <= after["cache_hit_rate"] <= 1.0
        assert after["model_loaded"] is True
