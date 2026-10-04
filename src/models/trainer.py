import json
import logging
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FraudModel:
    def __init__(
        self,
        model_type="xgboost",
        random_state=42,
        models_dir="models",
        reports_dir="reports",
        track_mlflow=True,
    ):
        if model_type not in {"xgboost", "random_forest"}:
            raise ValueError("Unsupported model_type")
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.track_mlflow = track_mlflow
        self.metrics = {}
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_engineer = None
        self.feature_columns = None

    def train(self, df):
        try:
            if not self.track_mlflow:
                raise ImportError("MLflow disabled for this run")
            # Import MLflow but don't fail if not configured
            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment("fraud-detection")
            use_mlflow = True
        except Exception as e:
            logger.info("MLflow not configured, continuing without tracking: %s", e)
            use_mlflow = False

        if use_mlflow:
            mlflow.start_run()

        try:
            # Feature engineering
            from src.features.engineer import FeatureEngineer

            # Split raw rows first. No held-out row may affect fitted feature statistics.
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=self.random_state, stratify=df["is_fraud"]
            )
            self.feature_engineer = FeatureEngineer()
            raw_train = train_df.drop(columns=["is_fraud"])
            raw_test = test_df.drop(columns=["is_fraud"])
            X_train = self.feature_engineer.fit_transform(raw_train)
            X_test = self.feature_engineer.transform(raw_test)
            self.feature_columns = list(X_train.columns)
            y_train, y_test = train_df["is_fraud"], test_df["is_fraud"]

            logger.info("Training set: %d samples", len(X_train))
            logger.info("Test set: %d samples", len(X_test))
            logger.info("Features: %d", len(self.feature_columns))

            # Class-imbalance handling: fraud is typically ~2-3% positive,
            # so an unweighted model will happily predict "not fraud" and
            # still score high on accuracy. We reweight the minority class
            # instead of resampling, so the reported metrics stay honest.
            n_pos = int(y_train.sum())
            n_neg = int(len(y_train) - n_pos)
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            logger.info(
                "Class balance in training set: %d positive / %d negative (scale_pos_weight=%.2f)",
                n_pos,
                n_neg,
                scale_pos_weight,
            )

            # Train model
            logger.info("Training %s model...", self.model_type)
            if self.model_type == "xgboost":
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                )
                self.model.fit(X_train, y_train)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    class_weight="balanced",
                )
                self.model.fit(X_train, y_train)

            # Evaluate
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            # Fraud is a rare-event problem (~2-3% positive class), where
            # ROC-AUC can look deceptively good. PR-AUC (average precision)
            # is more informative here since it focuses on the minority class.
            pr_auc_score = average_precision_score(y_test, y_pred_proba)

            baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
            baseline_proba = baseline.predict_proba(X_test)[:, 1]
            self.metrics = {
                "dataset": "caller supplied; see run metadata for provenance",
                "model_type": self.model_type,
                "random_state": self.random_state,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "test_prevalence": float(y_test.mean()),
                "roc_auc": float(auc_score),
                "average_precision": float(pr_auc_score),
                "baseline_roc_auc": float(roc_auc_score(y_test, baseline_proba)),
                "baseline_average_precision": float(
                    average_precision_score(y_test, baseline_proba)
                ),
                "classification_at_0_5": classification_report(
                    y_test, (y_pred_proba > 0.5).astype(int), output_dict=True, zero_division=0
                ),
                "threshold": 0.5,
                "features": self.feature_columns,
                "feature_statistics_fit_on": "training_rows_only",
                "brier_score": float(np.mean((y_pred_proba - y_test.to_numpy()) ** 2)),
            }
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.reports_dir / "evaluation.json").write_text(
                json.dumps(self.metrics, indent=2) + "\n"
            )

            # Log metrics
            if use_mlflow:
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("n_features", len(self.feature_columns))
                mlflow.log_param("scale_pos_weight", scale_pos_weight)
                mlflow.log_metric("auc_score", auc_score)
                mlflow.log_metric("pr_auc_score", pr_auc_score)
                mlflow.log_metric("test_size", len(X_test))
                mlflow.sklearn.log_model(self.model, "model")

            logger.info("Model trained successfully!")
            logger.info("ROC-AUC: %.4f", auc_score)
            logger.info("PR-AUC (average precision): %.4f", pr_auc_score)
            logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

            # Save locally
            self.save_model()

            return auc_score

        finally:
            if use_mlflow:
                mlflow.end_run()

    def predict(self, df):
        df_features = self.feature_engineer.transform(df)
        X = df_features[self.feature_columns]
        predictions = self.model.predict_proba(X)[:, 1]
        return predictions

    def save_model(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.models_dir / "fraud_model.pkl")
        joblib.dump(self.feature_engineer, self.models_dir / "feature_engineer.pkl")
        joblib.dump(self.feature_columns, self.models_dir / "feature_columns.pkl")
        logger.info("Model artifacts saved to models/")

    def load_model(self):
        self.model = joblib.load(self.models_dir / "fraud_model.pkl")
        self.feature_engineer = joblib.load(self.models_dir / "feature_engineer.pkl")
        self.feature_columns = joblib.load(self.models_dir / "feature_columns.pkl")
