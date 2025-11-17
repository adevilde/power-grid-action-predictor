import grid2op
import polars as pl
import numpy as np

from pathlib import Path
from typing import List, Tuple
from lightsim2grid import LightSimBackend
from grid2op.Observation import BaseObservation
from grid2op.Environment import Environment
from grid2op.Action import BaseAction
from tqdm import tqdm

# Paths for caching raw data
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEATURES = RAW_DIR / "df_features.parquet"
RAW_TARGETS = RAW_DIR / "df_targets.parquet"
RAW_ACTIONS = RAW_DIR / "all_actions.npy"


def extract_features(obs: BaseObservation) -> pl.DataFrame:
    """
    Note : The shapes are different between the features, so we store each feature
    vector in its own column.
    """
    def expand(prefix: str, values: np.ndarray) -> dict[str, list[float]]:
        return {f"{prefix}{idx + 1}": [val] for idx, val in enumerate(values)}

    column_values = {}
    column_values.update(expand("gen_p", obs.gen_p))
    column_values.update(expand("gen_q", obs.gen_q))
    column_values.update(expand("load_p", obs.load_p))
    column_values.update(expand("load_q", obs.load_q))
    column_values.update(expand("topo_vect", obs.topo_vect))
    column_values.update(expand("rho", obs.rho))

    return pl.DataFrame(column_values)


def create_realistic_observation(
    episode_count: int,
    env: Environment,
) -> list[BaseObservation]:
    """
    We create a list of realistic observation.
    This is a simple example of how to create a dataset from the environment.
    We break the temporal dependencies for simplicity.
    """

    list_obs = []
    for i in tqdm(range(episode_count)):
        obs = env.reset()
        list_obs.append(obs)
        # We go through each scenario, by doing the "nothing" action
        for _ in tqdm(range(env.chronics_handler.max_timestep())):
            obs, reward, done, info = env.step(env.action_space())
            if done:
                break
            list_obs.append(obs)

    return list_obs


def create_training_data(
    list_obs: list[BaseObservation],
    all_actions: list[BaseAction],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    We create the training data.

    For each observation we compute max rho on the lines after an action has been played.
    Under the hood, grid2op computes a power flow.
    (rho : how much the lines are loaded, between 0 and +inf but should be below 1 in normal operations)

    If this takes too long, you can reduce the number of actions (all_actions) or the number of observations (via episode count).
    Note : We are playing random actions, that might cause wrong situations. (like disconnecting a load)
    """

    df_features = []
    df_targets = []

    for _, obs in tqdm(enumerate(list_obs), total=len(list_obs)):
        action_score = []
        simulator = obs.get_simulator()

        for act in all_actions:
            sim_after_act = simulator.predict(act=act)
            n_obs = sim_after_act.current_obs
            action_score.append(n_obs.rho.max() if sim_after_act.converged else np.inf)

        df_targets.append(action_score)
        df_features.append(extract_features(obs))

    df_features = pl.concat(df_features)
    df_targets = pl.DataFrame(df_targets).transpose()
    df_targets = df_targets.rename(
        {col: f"action_{i}" for i, col in enumerate(df_targets.columns)}
    )

    return df_features, df_targets


def _run_simulation(
    env_name: str = "l2rpn_case14_sandbox",
    episode_count: int = 2,
    n_actions: int = 100,
    save_actions_path: Path = RAW_ACTIONS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Launch a grid2op environment, sample random actions, and gather the
    corresponding features/targets needed to train the predictor.

    Args:
        env_name: Name of the grid2op environment to instantiate.
        episode_count: Number of episodes to simulate when collecting observations.
        n_actions: Number of random actions to sample per observation (plus the do-nothing action).
        save_actions_path: Optional path where the sampled actions are persisted as vectors.

    Returns:
        Tuple of (df_features, df_targets) where df_features contains flattened
        observation data and df_targets stores the max post-action line loadings.
    """
    env = grid2op.make(env_name, backend=LightSimBackend(), n_busbar=3)

    all_actions = [env.action_space.sample() for _ in range(n_actions)]
    all_actions.append(env.action_space())

    if save_actions_path is not None:
        save_actions_path.parent.mkdir(parents=True, exist_ok=True)
        stacked_actions = np.vstack([act.to_vect() for act in all_actions])
        np.save(save_actions_path, stacked_actions)

    list_obs = create_realistic_observation(episode_count, env)
    df_features, df_targets = create_training_data(list_obs, all_actions)

    return df_features, df_targets


def generate_data(
    env_name: str = "l2rpn_case14_sandbox",
    episode_count: int = 2,
    n_actions: int = 100,
    force: bool = False
) -> None:
    """
    Create or reuse cached raw feature/target Parquet files for the specified
    grid2op environment, optionally forcing regeneration.

    Args:
        env_name: Name of the grid2op environment used for the simulation.
        episode_count: Number of episodes simulated when generating observations.
        n_actions: Number of random actions scored for each observation.
        force: Regenerate data even if cached Parquet files already exist.
    """
    if RAW_FEATURES.exists() and RAW_TARGETS.exists() and not force:
        print("Raw data already exists in data/raw/. Skipping simulation.")
        return

    print("Running grid simulation to create training data with")
    print(f"- {episode_count} episodes \n- {n_actions} actions")
    df_features, df_targets = _run_simulation(env_name, episode_count, n_actions)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(RAW_FEATURES)
    df_targets.write_parquet(RAW_TARGETS)
    print(f"Wrote raw Parquet files to {RAW_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", default="False", help="Force regeneration of raw data")
    parser.add_argument("--n_episodes", type=int, default=2, help="Number of episodes simulated when generating observations")
    parser.add_argument("--n_actions", type=int, default=100, help="Number of random actions scored for each observation")
    args = parser.parse_args()
    force_flag = str(args.force).lower() == "true"

    generate_data(episode_count=args.n_episodes, n_actions=args.n_actions, force=force_flag)
