from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = ROOT_DIR / "data" / "clean_dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

PREDICTION_RESULTS_PATH = RESULTS_DIR / "prediction_results.csv"
PREDICTION_PLOT_PATH = RESULTS_DIR / "predicted_vs_actual.png"
MODEL_METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_pv_power_model.joblib"

FEATURE_COLUMNS = [
    "irradiance",
    "ambient_temperature",
    "module_temperature",
    "humidity",
    "wind_speed",
    "hour",
    "day_of_year",
]
TARGET_COLUMN = "power_output"


def load_or_create_data() -> pd.DataFrame:
    # The progress-stage model should train from the cleaned dataset.
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {CLEAN_DATA_PATH}. Run `python src/preprocess_data.py` first."
        )

    return pd.read_csv(CLEAN_DATA_PATH)


def build_models() -> dict:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_prediction_plot(results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))

    for model_name, model_results in results.groupby("model"):
        plt.scatter(
            model_results["actual_power_output"],
            model_results["predicted_power_output"],
            alpha=0.35,
            s=14,
            label=model_name,
        )

    max_power = max(
        float(results["actual_power_output"].max()),
        float(results["predicted_power_output"].max()),
    )
    plt.plot([0, max_power], [0, max_power], color="black", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual power output (kW)")
    plt.ylabel("Predicted power output (kW)")
    plt.title("Predicted vs Actual PV Power Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_create_data()
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = []
    prediction_frames = []
    trained_models = {}

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        metrics.append(evaluate_model(model_name, y_test, y_pred))
        trained_models[model_name] = model

        model_predictions = X_test.copy()
        model_predictions["model"] = model_name
        model_predictions["actual_power_output"] = y_test.to_numpy()
        model_predictions["predicted_power_output"] = np.round(y_pred, 4)
        prediction_frames.append(model_predictions)

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    prediction_results = pd.concat(prediction_frames).sort_index()

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    prediction_results.to_csv(PREDICTION_RESULTS_PATH, index=False)
    save_prediction_plot(prediction_results)

    best_model_name = metrics_df.iloc[0]["model"]
    joblib.dump(
        {
            "model": trained_models[best_model_name],
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        BEST_MODEL_PATH,
    )

    print("Model metrics:")
    print(metrics_df.to_string(index=False))
    print(f"Saved prediction results: {PREDICTION_RESULTS_PATH}")
    print(f"Saved prediction plot: {PREDICTION_PLOT_PATH}")
    print(f"Saved model metrics: {MODEL_METRICS_PATH}")
    print(f"Saved best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
