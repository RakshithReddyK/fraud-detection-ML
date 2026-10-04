# Fraud Detection ML Service

A reproducible **synthetic-data** fraud-scoring service: XGBoost or Random Forest training, persisted preprocessing, FastAPI inference, optional Redis caching, baseline evaluation, and operational tests.

This is a portfolio system, not a bank-grade fraud decision engine. The measured threshold has low fraud precision; scores are not calibrated probabilities. See the [model card](MODEL_CARD.md).

## Run locally

```bash
git clone https://github.com/RakshithReddyK/fraud-detection-ML.git
cd fraud-detection-ML
poetry install
poetry run python scripts/evaluate_portfolio.py
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The evaluation command generates 50,000 synthetic rows with seed 42, splits 80/20, trains XGBoost, saves the three artifacts in `models/`, and writes `reports/evaluation.json`. It disables optional MLflow tracking. Existing `scripts/train.py` remains available for the CSV-based training workflow with MLflow when installed/configured.

```bash
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "amount":150.0,"merchant_risk_score":0.3,"days_since_last_transaction":1.0,
  "hour_of_day":14,"is_weekend":0,"num_transactions_today":3,"location_risk":0.2
}'
```

The prediction returns `fraud_probability` (an uncalibrated classifier score), `is_fraud` using the illustrative 0.5 threshold, actual request latency, a serving-bundle fingerprint, and a fresh timestamp. This service makes no financial actions.

## Evaluation that can be reproduced

[Captured report](reports/evaluation.json): 40,000 train / 10,000 test rows, 3% positive prevalence, generator seed 42, split/model seed 42. Feature statistics are fitted **only after splitting**, using training rows. The constant-prior baseline uses the same held-out rows.

| Metric | XGBoost | Constant-prior baseline |
|---|---:|---:|
| ROC-AUC | 0.7523 | 0.5000 |
| Average precision | 0.1470 | 0.0300 |

At threshold 0.5: fraud precision **0.0837**, recall **0.5967**, F1 **0.1468**. This operating point produces many false positives; it is not an approved business threshold. Threshold selection needs a separate validation set and explicit review costs. No threshold was tuned on the held-out test data.

The generator uses engineered relationships between observed features, latent variables, and random noise; it adjusts labels to a target prevalence. It is deliberately artificial. These scores demonstrate the pipeline on that generator, not performance on real fraud. No confidence intervals, temporal holdout, fairness audit, drift validation, or calibration study have been performed.

## Serving behavior

- `/health` checks process liveness and reports artifact/cache status. `/ready` returns 503 when the model bundle is unavailable.
- Artifacts load at startup, once per worker. The hash of model, feature transformer, and column list identifies the serving bundle.
- Requests reject extra fields, non-finite values, negative amounts/days, and out-of-range risk scores/hours. The feature order is persisted.
- Set `REDIS_URL` to enable caching. A cache outage preserves direct scoring. Connection/read timeouts are bounded; Redis is optional.
- Cache keys include the artifact fingerprint. Cache hits return the current request's timestamp and latency.
- `/metrics` exposes actual per-process counts, average prediction latency, and cache-hit rate. It does not aggregate across workers.
- Synchronous inference and optional Redis calls run in FastAPI's request thread pool.

## Latency and cost

[Captured HTTP benchmark](reports/latency.json): one Uvicorn worker, one persistent loopback client, 100 sequential repeated transactions after five warmups, Redis disabled. Measured p50 **5.49 ms**, p95 **6.94 ms**. This is not a concurrent-load test or a hosted latency guarantee.

Inference runs locally and invokes no hosted model API, so hosted-model cost is $0 per measured request. Hardware, service hosting, and optional Redis still have costs. No cloud bill or deployment cost was measured.

```bash
poetry run python scripts/benchmark.py
poetry run pytest -q
poetry run ruff check .
poetry run black --check .
```

Fourteen tests cover actual model/API inference, train-only preprocessing statistics, invalid/non-finite inputs, missing artifacts, cache outage recovery, fresh cache-hit metadata, and process metrics. Tests use temporary model/report directories and do not overwrite a developer's local trained artifacts. CI now targets the actual default branch, `master`.

## What changed in this production pass

The previous trainer computed feature statistics before splitting, despite the README describing train-only preprocessing. That leakage is fixed and regression-tested. Previously, an explicitly configured but unreachable Redis could clear an otherwise loaded model; it now falls back to direct inference. Cached responses previously reused old timing metadata and could cross model versions; they now use bundle-specific keys and fresh metadata.

## Layout

| Path | Responsibility |
|---|---|
| `src/data/generator.py` | Seeded synthetic transactions |
| `src/features/engineer.py` | Fitted feature statistics and transforms |
| `src/models/trainer.py` | Split, train, baseline evaluation, artifact persistence |
| `src/api/main.py` | Prediction API, readiness, cache and metrics |
| `scripts/evaluate_portfolio.py` | Reproduce the published synthetic evaluation |
| `scripts/benchmark.py` | Reproduce local HTTP timings |
| `tests/` | Model/serving regressions |

## Deployment boundary

The existing Dockerfile is retained; container build/runtime was not exercised in this change. The API has no built-in authentication or distributed rate limiter. Keep it on localhost or behind a gateway with TLS, identity and request limits. Load only trusted model artifacts: joblib/pickle is unsafe for untrusted files. Multi-worker metrics, safe model rollout, calibration, real-data validation and monitoring remain deployment work. No compliance certification or live production deployment is claimed.
