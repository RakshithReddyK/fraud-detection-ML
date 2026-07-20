import logging
import os

import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FraudModel:
    def __init__(self, model_type="xgboost", random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_engineer = None
        self.feature_columns = None

    def train(self, df):
        try:
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

            self.feature_engineer = FeatureEngineer()
            df_features = self.feature_engineer.fit_transform(df)

            # Prepare data
            target = "is_fraud"
            exclude_cols = [target]
            self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]

            X = df_features[self.feature_columns]
            y = df_features[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

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
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/fraud_model.pkl")
        joblib.dump(self.feature_engineer, "models/feature_engineer.pkl")
        joblib.dump(self.feature_columns, "models/feature_columns.pkl")
        logger.info("Model artifacts saved to models/")

    def load_model(self):
        self.model = joblib.load("models/fraud_model.pkl")
        self.feature_engineer = joblib.load("models/feature_engineer.pkl")
        self.feature_columns = joblib.load("models/feature_columns.pkl")
