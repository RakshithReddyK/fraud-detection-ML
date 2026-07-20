## Fraud Detection ML Pipeline

An end-to-end fraud detection service: synthetic data generation, feature
engineering, model training with MLflow tracking, and a FastAPI serving
layer with Redis caching — packaged with tests, CI, and Docker.

## Overview

* Synthetic transaction data generator (`FraudDataGenerator`) with a label
  that is deliberately **decoupled** from the observed features (see
  "Why synthetic labels aren't leaked" below) so reported metrics are honest.
* A feature engineering step (`FeatureEngineer`) that derives amount
  z-scores, risk combinations, time-of-day, and velocity features.
* A configurable model trainer (`FraudModel`, XGBoost or Random Forest) with
  class-imbalance handling and MLflow experiment tracking.
* A FastAPI service (`src/api/main.py`) exposing `/predict`, `/health`, and
  `/metrics`, with optional Redis response caching.
* Docker image, GitHub Actions CI (lint + test), and pre-commit hooks.

## Components

### 1. Synthetic Data Generator — `FraudDataGenerator`

Located in `src/data/generator.py`.

* Generates synthetic transaction-level data (amount, merchant risk score,
  time-of-day, transaction velocity, etc.).
* Default: 50,000 records, ~3% fraud rate (configurable).
* Saves the dataset to `data/transactions.csv`.

**Why synthetic labels aren't leaked**: an earlier version of this generator
derived `is_fraud` as a deterministic function of the same columns used as
model features (e.g. `amount > 95th percentile AND merchant_risk_score >
0.7`), which let the model memorize the labeling rule and report
near-perfect (~0.94+) AUC — a classic label-leakage artifact, not a real
signal. The label is now generated from a logistic combination where the
dominant terms are **latent variables that are never exposed as features**
(simulating unobserved real-world fraud signals like device fingerprints or
network-graph anomalies), plus independent noise. The observed columns still
contribute a real but weaker, non-deterministic signal. A trained model now
reports ROC-AUC around ~0.75 and PR-AUC clearly above the ~3% baseline —
realistic for a rare-event classification problem, and not an artifact of
leakage.

The generator makes this project fully self-contained; no external dataset
is required.

### 2. Feature Engineering — `FeatureEngineer`

Located in `src/features/engineer.py`. Fits statistics on training data
(amount mean/std, merchant risk mean) and derives features such as
`amount_z_score`, `amount_log`, `combined_risk`, `is_night`,
`is_business_hours`, and `high_velocity`. Persisted alongside the model so
the API applies identical transforms at inference time.

### 3. Model Trainer — `FraudModel`

Located in `src/models/trainer.py`.

* Trains an XGBoost or Random Forest classifier (`model_type="xgboost"` or
  `"random_forest"`).
* Handles class imbalance: `scale_pos_weight` for XGBoost, computed from the
  training split's positive/negative ratio, and `class_weight="balanced"`
  for Random Forest.
* Reports both **ROC-AUC** and **PR-AUC** (average precision). PR-AUC is
  reported because fraud is a ~2-3% positive class, where ROC-AUC alone can
  look deceptively good.
* Logs params/metrics/model to MLflow when available; degrades gracefully
  (with a log message) if MLflow isn't configured.
* Saves `models/fraud_model.pkl`, `models/feature_engineer.pkl`, and
  `models/feature_columns.pkl`.

```python
from src.models.trainer import FraudModel

model = FraudModel(model_type="xgboost", random_state=42)
auc_score = model.train(df)
```

### 4. Training Entrypoint — `scripts/train.py`

Orchestrates the pipeline end to end: generates data if missing, loads it,
trains `FraudModel`, logs ROC-AUC, and saves model artifacts.

### 5. Serving API — `src/api/main.py`

A FastAPI app with:

* `POST /predict` — scores a transaction, returns `fraud_probability`,
  `is_fraud`, `latency_ms`, and `model_version`. Uses Redis for response
  caching when `REDIS_URL` (or a local Redis instance) is available; falls
  back to computing predictions directly when Redis is unavailable.
* `GET /health` — reports whether the model/feature engineer loaded
  successfully and how many features are expected.
* `GET /metrics` — real, process-local serving metrics: total predictions
  served, average latency, and cache hit rate, tracked in-memory since
  process start (not hardcoded placeholders). For multi-worker/multi-instance
  deployments this would be swapped for a Prometheus exporter or similar so
  counters aggregate across processes.

### 6. Load Testing — `scripts/load_test.py`

A manual script (not a pytest test) that fires 100 requests at a locally
running API and reports average/P95 latency and throughput. Run it with the
API already up: `poetry run python scripts/load_test.py`.

## Project Structure

```
fraud-detection-ML/
├── .github/workflows/ci.yml    # lint + test on push/PR
├── data/
│   └── transactions.csv        # auto-generated if missing (gitignored)
├── models/
│   └── fraud_model.pkl         # saved trained model (gitignored)
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI serving layer
│   ├── data/
│   │   └── generator.py        # FraudDataGenerator
│   ├── features/
│   │   └── engineer.py         # FeatureEngineer
│   └── models/
│       └── trainer.py          # FraudModel
├── scripts/
│   ├── train.py                # full training pipeline entrypoint
│   └── load_test.py            # manual API load test
├── tests/
│   └── test_api.py             # pytest suite (trains a tiny model, hits the API)
├── Dockerfile
├── Makefile
├── pyproject.toml / poetry.lock
└── README.md
```

## Setup Instructions

This project uses [Poetry](https://python-poetry.org/) for dependency
management (there is no `requirements.txt`).

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/fraud-detection-ML.git
   cd fraud-detection-ML
   ```
2. Install dependencies (creates/uses a Poetry-managed virtualenv)
   ```bash
   poetry install
   ```
3. Run the training pipeline
   ```bash
   poetry run python scripts/train.py
   # or: make train
   ```
4. Start the API
   ```bash
   poetry run uvicorn src.api.main:app --reload --port 8000
   # or: make api
   ```
5. Run tests / lint
   ```bash
   poetry run pytest tests/ -v
   poetry run ruff check .
   poetry run black --check .
   # or: make test / make lint
   ```

### Example training output

```
Generated 50000 transactions with 3.00% fraud rate
Loading data...
Loaded 50000 transactions with 3.00% fraud rate
Training model...
Training set: 40000 samples
Test set: 10000 samples
Features: 14
Model trained successfully!
ROC-AUC: 0.7523
PR-AUC (average precision): 0.1470
Model saved to models/
```

(Exact numbers vary with generator/model settings and random seed.)

### Docker

```bash
make docker-build
make docker-run
```

The image is a multi-stage build, runs as a non-root user, and defines a
`HEALTHCHECK` against `/health`.

## Why This Project Matters

This project showcases practices used by real ML engineering teams:

* Honest metric reporting — labels are decoupled from features to avoid
  leakage, and PR-AUC is reported alongside ROC-AUC for a rare-event problem.
* Class-imbalance-aware training.
* A serving layer with caching, health checks, and real (non-stubbed)
  metrics.
* CI (lint + test on every push/PR) and pre-commit hooks.
* A Dockerized, non-root, health-checked deployment target.

## Future Enhancements

Ideas for extending this further:

* Real-time fraud scoring with a streaming source (Kafka/Kinesis) instead of
  synchronous request/response.
* Hyperparameter optimization (e.g. Optuna) integrated with the MLflow
  tracking already in place.
* Model monitoring / drift detection in production (e.g. comparing serving-
  time feature distributions against training-time distributions).
* Aggregate `/metrics` across multiple API workers/instances (e.g. via a
  Prometheus exporter) instead of the current in-process counters.
