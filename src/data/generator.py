import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FraudDataGenerator:
    def __init__(self, n_samples=10000, target_rate=0.03, seed=42):
        self.n_samples = int(n_samples)
        self.target_rate = float(target_rate)
        self.rng = np.random.default_rng(seed)

    def generate(self):
        # Generate realistic transaction data. These columns are also the
        # features the model trains on (see FeatureEngineer), so none of
        # them may deterministically define the label below.
        data = {
            "amount": self.rng.lognormal(mean=3.5, sigma=1.2, size=self.n_samples),
            "merchant_risk_score": self.rng.beta(2, 5, self.n_samples),
            "days_since_last_transaction": self.rng.exponential(2, self.n_samples),
            "hour_of_day": self.rng.integers(0, 24, self.n_samples),
            "is_weekend": self.rng.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            "num_transactions_today": self.rng.poisson(3, self.n_samples),
            "location_risk": self.rng.beta(2, 8, self.n_samples),
        }
        df = pd.DataFrame(data)

        # Artificial probabilistic labels combine engineered relationships in observed
        # features with hidden variables and random noise. Observed features are still
        # intentionally informative; these synthetic metrics do not establish realism
        # or generalization to actual transactions. Target prevalence is adjusted below.
        latent_fraud_ring = self.rng.choice(
            [0, 1], self.n_samples, p=[1 - self.target_rate, self.target_rate]
        )
        latent_risk = self.rng.normal(0, 1, self.n_samples)

        fraud_logit = (
            -4.8
            + 2.8 * (df["amount"] > df["amount"].quantile(0.95))
            + 3.0 * (df["merchant_risk_score"] > 0.7)
            + 1.6 * df["hour_of_day"].between(0, 6)
            + 1.8 * (df["num_transactions_today"] > 10)
            + 1.0 * latent_fraud_ring
            + 0.25 * latent_risk
            + self.rng.normal(0, 0.25, self.n_samples)
        )
        fraud_probability = 1.0 / (1.0 + np.exp(-fraud_logit))
        df["is_fraud"] = (self.rng.random(self.n_samples) < fraud_probability).astype(int)

        # Adjust to target fraud rate (~3%)
        target_fraud = int(round(self.n_samples * self.target_rate))
        current_fraud = int(df["is_fraud"].sum())

        if current_fraud < target_fraud:
            need = target_fraud - current_fraud
            zero_idx = df.index[df["is_fraud"] == 0].to_numpy()
            k = min(need, zero_idx.size)
            if k > 0:
                flip = self.rng.choice(zero_idx, size=k, replace=False)
                df.loc[flip, "is_fraud"] = 1
        elif current_fraud > target_fraud:
            # Optional: trim down to target to keep dataset balanced to spec
            excess = current_fraud - target_fraud
            one_idx = df.index[df["is_fraud"] == 1].to_numpy()
            k = min(excess, one_idx.size)
            if k > 0:
                flip = self.rng.choice(one_idx, size=k, replace=False)
                df.loc[flip, "is_fraud"] = 0

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generator = FraudDataGenerator(n_samples=50000, target_rate=0.03, seed=42)
    df = generator.generate()

    # Ensure output directory exists
    Path("data").mkdir(parents=True, exist_ok=True)

    df.to_csv("data/transactions.csv", index=False)
    fraud_rate_pct = df["is_fraud"].mean() * 100
    logger.info("Generated %d transactions with %.2f%% fraud rate", len(df), fraud_rate_pct)
