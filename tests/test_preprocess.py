import ast
from pathlib import Path
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


def test_build_preprocessed_dataset_uses_cached_file(tmp_path):
    cached_df = pl.DataFrame({"f1": [1.0, 2.0], "action_0": [0.5, 0.6]})
    dataset_path = tmp_path / "dataset.parquet"
    cached_df.write_parquet(dataset_path)

    with patch.object(pp, "CACHED_DATASET", dataset_path), patch.object(
        pp, "load_raw"
    ) as mock_load:
        result = pp.build_preprocessed_dataset(force=False)

    assert mock_load.call_count == 0
    assert result.to_dict(as_series=False) == cached_df.to_dict(as_series=False)
