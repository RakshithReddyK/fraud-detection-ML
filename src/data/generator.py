import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class FraudDataGenerator:
    def __init__(self, n_samples=10000, target_rate=0.03, seed=42):
        self.n_samples = int(n_samples)
        self.target_rate = float(target_rate)
        self.rng = np.random.default_rng(seed)

    def generate(self):
        # Generate realistic transaction data
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

        # Heuristic fraud probability
        fraud_probability = (
            (df["amount"] > df["amount"].quantile(0.95)) * 0.3
            + (df["merchant_risk_score"] > 0.7) * 0.4
            + (df["hour_of_day"].between(0, 6)) * 0.2
            + (df["num_transactions_today"] > 10) * 0.3
        )

        df["is_fraud"] = (fraud_probability > self.rng.random(self.n_samples)).astype(int)

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
    generator = FraudDataGenerator(n_samples=50000, target_rate=0.03, seed=42)
    df = generator.generate()

    # Ensure output directory exists
    Path("data").mkdir(parents=True, exist_ok=True)

    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions with {df['is_fraud'].mean():.2%} fraud rate")
