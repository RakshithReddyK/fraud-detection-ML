import pandas as pd
import numpy as np
from typing import Dict, Iterable

class FeatureEngineer:
    REQUIRED_COLS: Iterable[str] = (
        "amount",
        "merchant_risk_score",
        "location_risk",
        "hour_of_day",
        "num_transactions_today",
    )

    def __init__(self, std_epsilon: float = 1e-8):
        self.feature_stats: Dict[str, float] = {}
        self.std_epsilon = float(std_epsilon)
        self._fitted = False

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Calculate statistics on training data."""
        self._validate_columns(df)

        amount_mean = float(df["amount"].mean())
        # ddof=0 for population std, and guard with epsilon
        amount_std = float(df["amount"].std(ddof=0))
        if not np.isfinite(amount_std) or amount_std < self.std_epsilon:
            amount_std = self.std_epsilon

        self.feature_stats = {
            "amount_mean": amount_mean,
            "amount_std": amount_std,
            "merchant_risk_mean": float(df["merchant_risk_score"].mean()),
        }
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features on a copy of df."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit().")

        self._validate_columns(df)
        out = df.copy()

        # Ensure hour_of_day is integer for comparisons
        out["hour_of_day"] = out["hour_of_day"].astype(int, errors="ignore")

        # Amount features
        out["amount_z_score"] = (
            (out["amount"] - self.feature_stats["amount_mean"])
            / self.feature_stats["amount_std"]
        )
        out["amount_log"] = np.log1p(out["amount"].clip(lower=0))

        # Risk combinations
        out["combined_risk"] = out["merchant_risk_score"] * out["location_risk"]
        out["high_amount_late_night"] = (
            (out["amount"] > 500)
            & (out["hour_of_day"].between(0, 6, inclusive="both"))
        ).astype(int)

        # Time features
        # Wraparound night: 22–23 OR 0–6
        out["is_night"] = (
            (out["hour_of_day"] >= 22) | (out["hour_of_day"] <= 6)
        ).astype(int)
        out["is_business_hours"] = out["hour_of_day"].between(
            9, 17, inclusive="both"
        ).astype(int)

        # Velocity features
        out["high_velocity"] = (out["num_transactions_today"] > 5).astype(int)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
