from unittest.mock import patch

import numpy as np
import pytest

import power_grid_pred.model as ml


class DummyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean_ = y.mean(axis=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        mean = self.mean_ if self.mean_ is not None else np.zeros(1, dtype=float)
        return np.tile(mean, (X.shape[0], 1))


def test_train_model_and_load(tmp_path):
    rng = np.random.default_rng(0)
    X_train = rng.random((6, 3))
    X_test = rng.random((2, 3))
    y_train = rng.random((6, 2))
    y_test = rng.random((2, 2))

    model_path = tmp_path / "model.joblib"

    with patch.object(
        ml, "train_test_split_dataset", return_value=(X_train, X_test, y_train, y_test)
    ), patch.object(ml, "MODEL_PATH", model_path):
        with patch.object(ml, "RandomForestRegressor", DummyRegressor):
            metrics = ml.train_model()

    assert set(metrics) == {"rmse", "mae", "r2"}

    loaded = ml.load_model()
    assert hasattr(loaded, "predict")


def test_load_model_missing(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with patch.object(ml, "MODEL_PATH", missing_path):
        with pytest.raises(FileNotFoundError):
            ml.load_model()
