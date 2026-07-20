#!/usr/bin/env python3
import logging
import os
import sys

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.models.trainer import FraudModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Check if data exists
    if not os.path.exists("data/transactions.csv"):
        logger.info("No data found. Running data generator first...")
        from src.data.generator import FraudDataGenerator

        generator = FraudDataGenerator(n_samples=50000)
        df = generator.generate()
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/transactions.csv", index=False)
        logger.info("Generated %d transactions", len(df))

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv("data/transactions.csv")
    fraud_rate_pct = df["is_fraud"].mean() * 100
    logger.info("Loaded %d transactions with %.2f%% fraud rate", len(df), fraud_rate_pct)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Train model
    logger.info("Training model...")
    model = FraudModel()
    auc_score = model.train(df)

    logger.info("Training complete!")
    logger.info("ROC-AUC score: %.4f", auc_score)
    logger.info("Model saved to models/")


if __name__ == "__main__":
    main()
