# src/power_grid_pred/visualization.py

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .generate_data import DATA_DIR
from .preprocess import build_preprocessed_dataset, train_test_split_dataset
from .model import load_model

FIG_DIR = DATA_DIR / "figures"

def _get_dataset():
    """Helper to load the processed, capped dataset as pandas."""
    df = build_preprocessed_dataset()
    return df.to_pandas()


def _get_action_cols(pdf):
    """Return the list of action column names (action_*) present in the dataset."""
    action_cols = [c for c in pdf.columns if c.startswith("action_")]
    if not action_cols:
        raise ValueError("No action_ columns found in dataset for visualization.")
    return action_cols


def _get_prediction_context():
    """
    Loads the dataset, splits it, loads the trained model,
    and computes predictions on the test set.

    Returns a dict with:
        X_train, X_test, y_train, y_test, y_pred, model
    """
    X_train, X_test, y_train, y_test = train_test_split_dataset()
    model = load_model()
    y_pred = model.predict(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
    }


def plot_best_rho_per_state(show: bool = False) -> Path:
    """
    For each grid state (row), compute the minimum rho over all actions.
    This answers: "How good can we do if we pick the best action?"

    Returns the path to the saved PNG.
    """
    pdf = _get_dataset()
    action_cols = _get_action_cols(pdf)

    # Best-case rho per state (lower is better)
    best_rho_per_state = pdf[action_cols].min(axis=1)

    out_path = FIG_DIR / "best_rho_per_state_histogram.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    best_rho_per_state.hist(bins=30, edgecolor="white")
    plt.xlabel("Best rho over all actions for a state")
    plt.ylabel("Observation Count")
    plt.title("Distribution of best-case line loading (rho) per grid state")
    plt.tight_layout()
    plt.savefig(out_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved histogram of best rho per state to {out_path} \n")
    return out_path


def plot_action_rho_boxplot(max_actions: int = 20, show: bool = False) -> Path:
    """
    Plot a boxplot of rho values for a subset of actions (action_0, action_1, ...).

    This shows which actions tend to be more risky / overloaded across all states.

    Args:
        max_actions (int): Max number of action_* columns to display (for readability).
        show (bool): If True, display interactively; otherwise save and close.

    Returns (Path): Path to the saved PNG file.
    """
    pdf = _get_dataset()
    action_cols = _get_action_cols(pdf)

    # Randomly pick up to max_actions action columns
    n = min(max_actions, len(action_cols))
    action_cols = list(np.random.choice(action_cols, size=n, replace=False))

    data = [pdf[c].values for c in action_cols]

    out_path = FIG_DIR / "action_rho_boxplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=[c for c in action_cols])
    plt.xlabel("Action index")
    plt.ylabel("Rho")
    plt.title("Distribution of rho per action (capped non-convergent flows)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved action rho boxplot to {out_path} \n")
    return out_path


def plot_predictions_for_action(
    show: bool = False,
) -> Path:
    """
    Scatter plot of predicted vs. true rho for one randomly selected action.

    Useful to eyeball how well the model lines up with ground truth for 
    individual actions instead of aggregate metrics.
    """
    ctx = _get_prediction_context()
    y_true = ctx["y_test"]
    y_pred = ctx["y_pred"]

    # pick a single random action index (column) from the predictions
    action_idx = np.random.randint(0, y_true.shape[1])

    # y_true is a pandas DataFrame while y_pred is a numpy array, so slice them accordingly
    y_true_action = y_true[f"action_{action_idx}"].to_numpy()
    y_pred_action = y_pred[:, action_idx]

    out_path = FIG_DIR / f"pred_vs_true_action_{action_idx}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_true_action, y_pred_action, alpha=0.4, s=10)

    # 45-degree reference line
    mins = min(y_true_action.min(), y_pred_action.min())
    maxs = max(y_true_action.max(), y_pred_action.max())
    plt.plot([mins, maxs], [mins, maxs], "k--", linewidth=1)

    plt.xlabel(f"True rho (action_{action_idx})")
    plt.ylabel(f"Predicted rho (action_{action_idx})")
    plt.title(f"Prediction scatter for action_{action_idx}")
    plt.tight_layout()
    plt.savefig(out_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved preditions for action_{action_idx} at {out_path} \n")
    return out_path


def plot_rmse_per_action(max_actions: int = 20, show: bool = False) -> Path:
    """
    Generate a bar chart of per-action RMSE on the test set.

    Args:
        max_actions: cap the number of actions shown (random subset) for readability.
        show: when True display interactively, otherwise save a PNG and close.
    """
    ctx = _get_prediction_context()
    y_true = ctx["y_test"]
    y_pred = ctx["y_pred"]

    # Compute RMSE per action
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean(axis=0))

    # Choose actions to visualize (random subset or first N)
    n_actions = rmse.shape[0]
    k = min(n_actions, max_actions)
    action_idxs = np.random.choice(n_actions, size=k, replace=False)

    # Values and labels
    values = rmse.iloc[action_idxs].to_numpy()
    labels = [f"action_{i}" for i in action_idxs]

    # Use dense x-positions 0..k-1
    x = np.arange(k)

    out_path = FIG_DIR / "rmse_per_action.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Action")
    plt.ylabel("RMSE")
    plt.title("RMSE per action on test set")
    plt.tight_layout()
    plt.savefig(out_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved RMSE per action bar chart to {out_path}\n")
    return out_path


if __name__ == "__main__":
    # Generate plots when run as a script
    print("Plotting the distribution of best-case line loading (rho) per grid state")
    plot_best_rho_per_state()
    print("Plotting the distribution of rho per action (capped non-convergent flows)")
    plot_action_rho_boxplot()
    print("Plotting the prediction scatter for a random action")
    plot_predictions_for_action()
    print("Plotting RMSE per action on test set")
    plot_rmse_per_action()
