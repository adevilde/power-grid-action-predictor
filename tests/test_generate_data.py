from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

import power_grid_pred.generate_data as gd


@pytest.fixture
def raw_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    features_path = raw_dir / "df_features.parquet"
    targets_path = raw_dir / "df_targets.parquet"

    with ExitStack() as stack:
        stack.enter_context(patch.object(gd, "RAW_DIR", raw_dir))
        stack.enter_context(patch.object(gd, "RAW_FEATURES", features_path))
        stack.enter_context(patch.object(gd, "RAW_TARGETS", targets_path))
        yield raw_dir, features_path, targets_path


def _obs_with_predictions(pred_sequence):
    iterator = iter(pred_sequence)

    def predict(act):
        rho_vals, converged = next(iterator)
        return SimpleNamespace(
            current_obs=SimpleNamespace(rho=np.array(rho_vals, dtype=float)),
            converged=converged,
        )

    simulator = SimpleNamespace(predict=predict)
    return SimpleNamespace(
        gen_p=np.array([1.0]),
        gen_q=np.array([0.1]),
        load_p=np.array([0.5]),
        load_q=np.array([0.2]),
        topo_vect=np.array([0, 1]),
        rho=np.array([0.3]),
        get_simulator=lambda: simulator,
    )


def test_extract_features_builds_prefixed_columns():
    obs = SimpleNamespace(
        gen_p=np.array([1.0, 2.0]),
        gen_q=np.array([0.1]),
        load_p=np.array([3.0]),
        load_q=np.array([4.0, 5.0]),
        topo_vect=np.array([0, 1, 0]),
        rho=np.array([0.4]),
    )

    df = gd.extract_features(obs)

    assert df.shape == (1, 10)
    assert df["gen_p1"].item() == 1.0
    assert df["rho1"].item() == 0.4


def test_create_training_data_handles_non_converged_actions():
    obs = _obs_with_predictions(
        [
            ([0.2, 0.4], True),
            ([1.5], False),
        ]
    )
    df_features, df_targets = gd.create_training_data(
        list_obs=[obs],
        all_actions=["a", "b"],
    )

    assert df_features.height == 1
    assert df_targets.columns == ["action_0", "action_1"]
    values = df_targets.to_dict(as_series=False)
    assert values["action_0"][0] == 0.4
    assert values["action_1"][0] == float("inf")


def test_create_realistic_observation_collects_until_done():
    timeline = ["reset", "step1", "step2"]
    state = {"idx": 0}

    def reset():
        state["idx"] = 0
        return timeline[0]

    def step(action):
        state["idx"] += 1
        obs = timeline[state["idx"]]
        done = state["idx"] == len(timeline) - 1
        return obs, 0.0, done, {}

    handler = SimpleNamespace(max_timestep=lambda: 2)
    env = SimpleNamespace(
        reset=reset,
        step=step,
        action_space=lambda: "noop",
        chronics_handler=handler,
    )

    with patch.object(gd, "tqdm", lambda iterable, *args, **kwargs: iterable):
        observations = gd.create_realistic_observation(episode_count=1, env=env)

    assert observations == ["reset", "step1"]


@patch.object(gd, "_run_simulation")
def test_generate_data_force_true(mock_run, raw_cache):
    fake_features = pl.DataFrame({"feature": [1.0]})
    fake_targets = pl.DataFrame({"action_0": [0.1]})

    _, features_path, targets_path = raw_cache

    mock_run.return_value = (fake_features, fake_targets)
    gd.generate_data(force=True)

    assert mock_run.call_count == 1
    assert features_path.exists()
    assert targets_path.exists()


@patch.object(gd, "_run_simulation")
def test_generate_data_force_false_skips_when_cached(mock_run, raw_cache):
    raw_dir, features_path, targets_path = raw_cache
    raw_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"feature": [1.0]}).write_parquet(features_path)
    pl.DataFrame({"action_0": [0.9]}).write_parquet(targets_path)

    gd.generate_data(force=False)

    assert mock_run.call_count == 0


@patch.object(gd, "create_training_data")
@patch.object(gd, "create_realistic_observation")
@patch.object(gd, "LightSimBackend")
@patch.object(gd.grid2op, "make")
def test_run_simulation_uses_grid2op(mock_make, mock_backend, mock_create_obs, mock_create_training, tmp_path):
    def make_action(value):
        return SimpleNamespace(to_vect=lambda value=value: np.array([value], dtype=float))

    action_space = Mock()
    action_space.sample.side_effect = [
        make_action(0.1),
        make_action(0.2),
    ]
    action_space.return_value = make_action(-1.0)
    env_instance = SimpleNamespace(action_space=action_space)

    mock_make.return_value = env_instance
    mock_backend.return_value = "backend"

    mock_create_obs.return_value = ["obs"]
    mock_create_training.return_value = (
        pl.DataFrame({"feature": [0.1]}),
        pl.DataFrame({"action_0": [0.5]}),
    )

    actions_path = tmp_path / "actions.npy"
    df_features, df_targets = gd._run_simulation(
        env_name="foo",
        episode_count=3,
        n_actions=2,
        save_actions_path=actions_path,
    )

    assert mock_make.call_count == 1
    args, kwargs = mock_make.call_args
    assert args[0] == "foo"
    assert kwargs["backend"] == "backend"
    assert kwargs["n_busbar"] == 3
    mock_create_obs.assert_called_once_with(3, env_instance)
    args, _ = mock_create_training.call_args
    assert args[0] == ["obs"]
    assert len(args[1]) == 3  # n_actions + noop
    assert action_space.sample.call_count == 2
    assert action_space.call_count == 1
    assert df_features.shape == (1, 1)
    assert df_targets.shape == (1, 1)

    assert actions_path.exists()
    saved = np.load(actions_path)
    assert saved.shape == (3, 1)
    assert np.allclose(saved[:, 0], [0.1, 0.2, -1.0])
