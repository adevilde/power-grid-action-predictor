from unittest.mock import patch

import polars as pl
import pytest

import power_grid_pred.preprocess as pp


def test_build_dataset_replaces_non_finite_targets(tmp_path):
    df_features = pl.DataFrame({"f1": [1.0, 2.0, 3.0]})
    df_targets = pl.DataFrame(
        {
            "action_0": [1.0, 2.0, 3.0],
            "action_1": [float("inf"), float("-inf"), float("nan")],
        }
    )

    dataset_path = tmp_path / "dataset.parquet"

    with patch.object(pp, "CACHED_DATASET", dataset_path), patch.object(
        pp, "load_raw", return_value=(df_features, df_targets)
    ):
        df = pp.build_preprocessed_dataset(force=True)

    assert dataset_path.exists()
    assert df.height == 3
    assert df["action_0"].to_list() == [1.0, 2.0, 3.0]
    assert df["action_1"].to_list() == [8.0, 8.0, 8.0]  # rho_cap = max_finite(3) + 5


def test_train_test_split_shapes(tmp_path):
    df = pl.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "action_0": [0.5, 0.6, 0.7, 0.8],
            "action_1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    dataset_path = tmp_path / "dataset.parquet"
    df.write_parquet(dataset_path)

    with patch.object(pp, "CACHED_DATASET", dataset_path):
        X_train, X_test, y_train, y_test = pp.train_test_split_dataset()

    assert X_train.shape[0] + X_test.shape[0] == 4
    assert y_train.ndim == 2
    assert y_test.ndim == 2


def test_load_raw_reads_parquet(tmp_path):
    features_path = tmp_path / "f.parquet"
    targets_path = tmp_path / "t.parquet"
    df_features = pl.DataFrame({"f": [1.0, 2.0]})
    df_targets = pl.DataFrame({"action_0": [0.1, 0.2]})

    df_features.write_parquet(features_path)
    df_targets.write_parquet(targets_path)

    with patch.object(pp, "RAW_FEATURES", features_path), patch.object(
        pp, "RAW_TARGETS", targets_path
    ):
        loaded_features, loaded_targets = pp.load_raw()

    assert loaded_features.to_dict(as_series=False) == df_features.to_dict(
        as_series=False
    )
    assert loaded_targets.to_dict(as_series=False) == df_targets.to_dict(
        as_series=False
    )


def test_load_raw_missing_files(tmp_path):
    features_path = tmp_path / "missing_features.parquet"
    targets_path = tmp_path / "missing_targets.parquet"

    with patch.object(pp, "RAW_FEATURES", features_path), patch.object(
        pp, "RAW_TARGETS", targets_path
    ):
        with pytest.raises(FileNotFoundError):
            pp.load_raw()


def test_build_preprocessed_dataset_errors_when_no_targets(tmp_path):
    df_features = pl.DataFrame({"f1": [1.0, 2.0]})
    df_targets = pl.DataFrame()
    dataset_path = tmp_path / "dataset.parquet"

    with patch.object(pp, "CACHED_DATASET", dataset_path), patch.object(
        pp, "load_raw", return_value=(df_features, df_targets)
    ):
        with pytest.raises(ValueError):
            pp.build_preprocessed_dataset(force=True)
