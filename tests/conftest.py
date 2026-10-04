import importlib

import pytest


# ---- Helpers to build a tiny model bundle for tests ----
@pytest.fixture(scope="session")
def tmp_models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models_bundle")
    return d


@pytest.fixture(scope="session")
def build_artifacts(tmp_models_dir):
    """
    Train a very small model and write the three expected artifacts:
    fraud_model.pkl, feature_engineer.pkl, feature_columns.pkl
    """
    # Import here to avoid importing app prematurely
    from src.data.generator import FraudDataGenerator
    from src.models.trainer import FraudModel

    # Small dataset for speed
    df = FraudDataGenerator(n_samples=3000, target_rate=0.03, seed=123).generate()

    # Train a quick baseline (no xgboost required in CI)
    model = FraudModel(
        model_type="random_forest",
        random_state=7,
        models_dir=tmp_models_dir,
        reports_dir=tmp_models_dir / "reports",
        track_mlflow=False,
    )
    model.train(df)

    return tmp_models_dir


@pytest.fixture
def app_with_artifacts(monkeypatch, build_artifacts):
    """
    Set MODELS_DIR env var BEFORE importing the FastAPI app module,
    so its startup uses our temp artifacts.
    """
    # 1) Point API to our temp models dir
    monkeypatch.setenv("MODELS_DIR", str(build_artifacts))
    # Optional: ensure Redis is not required in CI
    monkeypatch.delenv("REDIS_URL", raising=False)

    # 2) Import (or reload) the app module AFTER env is set
    #    so that module-level constants pick up the new MODELS_DIR.
    from src.api import main as app_module

    importlib.reload(app_module)  # pick up new env
    return app_module.app  # FastAPI instance
