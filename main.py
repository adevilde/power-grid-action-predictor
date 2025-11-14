import grid2op
import polars as pl
import numpy as np

from lightsim2grid import LightSimBackend
from grid2op.Observation import BaseObservation
from grid2op.Environment import Environment
from grid2op.Action import BaseAction
from tqdm import tqdm


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

    return df_features, df_targets


if __name__ == "__main__":
    # Create a fake power grid environment. This comes with built in data.
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend(), n_busbar=3)

    # if you want to vizualise the grid we are working with (requires matplotlib) :
    # env.render()

    # Episode refer here to different scenarios (Like two different forecasts for instance)
    episode_count = 2
    n_actions = 100

    # take n_actions random actions and add the do nothing action
    all_actions = [env.action_space.sample() for _ in range(n_actions)]
    all_actions.append(env.action_space())

    print("Example of pretty print for an action ;", all_actions[0])

    list_obs = create_realistic_observation(episode_count, env)
    df_features, df_targets = create_training_data(list_obs, all_actions)

    print(df_features)
    print(df_targets)

    df_targets.write_parquet("df_targets.parquet")
    df_features.write_parquet("df_features.parquet")
