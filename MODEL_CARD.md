# Fraud model card

**Review date:** 2026-10-04. **Model:** XGBoost classifier trained by `scripts/evaluate_portfolio.py`.

**Purpose:** demonstrate a reproducible train → persist → serve workflow on synthetic transactions. The output is an uncalibrated fraud score. The API does not approve, deny, or block payments.

**Data:** 50,000 generated transactions, 3% positive labels, generator seed 42. Eighty percent is used for training and 20% for a stratified test split, random state 42. Seven observed input fields become fourteen features. The generator explicitly links some observed features to labels, adds latent/noisy terms, and adjusts labels to the target prevalence. It is not a sample of real customers or fraud behavior.

**Preprocessing:** split before fitting amount statistics; persist the training-fitted transformer and ordered columns. Regression tests verify that held-out rows do not influence the fitted mean.

**Evaluation:** on 10,000 held-out synthetic rows, ROC-AUC 0.7523 and average precision 0.1470; constant-prior baseline ROC-AUC 0.5 and AP 0.03. At threshold 0.5, fraud precision 0.0837, recall 0.5967 and F1 0.1468. Exact values, dependency versions, features and class counts are in `reports/evaluation.json`. A single split has no uncertainty estimate and is not a temporal or external validation.

**Known failures:** high false-positive burden, uncalibrated scores, no threshold cost optimization, no subgroup/fairness analysis, no real-data or drift validation. Merchant/location risk inputs are synthetic numeric fields with no real-world feature contract. Random splitting does not test future drift or entity leakage in a real transaction stream. Do not infer financial effectiveness from these results.

**Serving:** validated FastAPI inputs; trusted local artifacts loaded at startup; bundle fingerprints in predictions/cache keys; optional Redis with graceful outage behavior; readiness and actual process metrics. Recorded one-worker HTTP p95 6.94 ms for 100 sequential repeated requests, excluding training and cloud/network deployment.

**Use restrictions:** local portfolio and engineering experiments only. A real fraud workflow requires authorized real data, temporal/entity-aware validation, calibration, threshold selection on validation data, business review, monitoring, identity, audit controls and human oversight. No regulatory or banking readiness claim is made.

**Reproduce:** `poetry run python scripts/evaluate_portfolio.py`, `poetry run python scripts/benchmark.py`, and `poetry run pytest -q`. Model artifacts are regenerated locally and are not committed. Hosted inference cost is zero for this local model; infrastructure cost was not measured.
