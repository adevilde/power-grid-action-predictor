import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from .preprocess import train_test_split_dataset, PROC_DIR

MODEL_PATH = PROC_DIR/"model.joblib"

def train_model() -> dict:
    """Train a RandomForestRegressor and persist it alongside evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split_dataset()

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Multi-output: compute metrics on flattened arrays for simplicity 
    rmse = root_mean_squared_error(y_test, y_pred) 
    mae = mean_absolute_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print("Model trained. Metrics:")
    for k,v in metrics.items():
        print(f" {k}: {round(v, 3)}")

    return metrics


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not trained yet. Run train_model() first."
        )
    
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train_model()
