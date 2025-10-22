import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

class FraudModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
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
            print(f"MLflow not configured, continuing without tracking: {e}")
            use_mlflow = False
        
        if use_mlflow:
            mlflow.start_run()
        
        try:
            # Feature engineering
            from src.features.engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
            df_features = self.feature_engineer.fit_transform(df)
            
            # Prepare data
            target = 'is_fraud'
            exclude_cols = [target]
            self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]
            
            X = df_features[self.feature_columns]
            y = df_features[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            print(f"Features: {len(self.feature_columns)}")
            
            # Train model
            if self.model_type == 'xgboost':
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'  # Add this to suppress warning
                )
                print(f"Training {self.model_type} model...")
                # Simple fit without early_stopping_rounds
                self.model.fit(X_train, y_train)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                print(f"Training {self.model_type} model...")
                self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            if use_mlflow:
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("n_features", len(self.feature_columns))
                mlflow.log_metric("auc_score", auc_score)
                mlflow.log_metric("test_size", len(X_test))
                mlflow.sklearn.log_model(self.model, "model")
            
            print(f"\n✅ Model trained successfully!")
            print(f"📊 AUC Score: {auc_score:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
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
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/fraud_model.pkl')
        joblib.dump(self.feature_engineer, 'models/feature_engineer.pkl')
        joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
        print(f"💾 Model artifacts saved to models/")
    
    def load_model(self):
        self.model = joblib.load('models/fraud_model.pkl')
        self.feature_engineer = joblib.load('models/feature_engineer.pkl')
        self.feature_columns = joblib.load('models/feature_columns.pkl')
