"""Generate a reproducible synthetic-data evaluation; no external data or MLflow."""

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import FraudDataGenerator
from src.models.trainer import FraudModel


def main():
    df = FraudDataGenerator(n_samples=50000, target_rate=0.03, seed=42).generate()
    model = FraudModel(model_type="xgboost", random_state=42, track_mlflow=False)
    model.train(df)
    report = {
        **model.metrics,
        "dataset": "FraudDataGenerator synthetic transactions",
        "generator_seed": 42,
        "rows": len(df),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost"]},
        "scope": (
            "Synthetic holdout; not real-world fraud performance. "
            "No calibration or subgroup audit."
        ),
    }
    Path("reports/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "roc_auc",
                    "average_precision",
                    "baseline_average_precision",
                    "test_prevalence",
                    "classification_at_0_5",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
