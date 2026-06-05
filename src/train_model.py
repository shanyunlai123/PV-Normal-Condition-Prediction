from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
WEATHER_DATA_DIR = DATA_DIR / "weather_datasets"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

CLEAN_DATA_PATH = DATA_DIR / "clean_dataset.csv"
ALL_WEATHER_PATH = WEATHER_DATA_DIR / "all_weather_dataset.csv"
SUNNY_PATH = WEATHER_DATA_DIR / "sunny_dataset.csv"
CLOUDY_PATH = WEATHER_DATA_DIR / "cloudy_dataset.csv"

PREDICTION_RESULTS_PATH = RESULTS_DIR / "prediction_results.csv"
PREDICTION_PLOT_PATH = RESULTS_DIR / "predicted_vs_actual.png"
MODEL_COMPARISON_PLOT_PATH = RESULTS_DIR / "model_comparison.png"
MODEL_METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
BEST_MODELS_PATH = RESULTS_DIR / "best_models_by_dataset.csv"
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


def load_training_datasets() -> dict:
    """Load all available datasets for scenario-based model comparison."""
    if ALL_WEATHER_PATH.exists() and SUNNY_PATH.exists() and CLOUDY_PATH.exists():
        return {
            "all_weather": pd.read_csv(ALL_WEATHER_PATH),
            "sunny": pd.read_csv(SUNNY_PATH),
            "cloudy": pd.read_csv(CLOUDY_PATH),
        }

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {CLEAN_DATA_PATH}. Run `python src/preprocess_data.py` first."
        )

    return {"all_weather": pd.read_csv(CLEAN_DATA_PATH)}


def build_models() -> dict:
    """Build different regression models for comparison."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "KNN Regressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=8)),
            ]
        ),
        "SVR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=50, epsilon=0.2)),
            ]
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
    }


def evaluate_model(dataset_name: str, model_name: str, rows: int, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate model metrics for one dataset/model pair."""
    return {
        "dataset": dataset_name,
        "model": model_name,
        "rows": int(rows),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_prediction_plot(results: pd.DataFrame) -> None:
    """Plot best-model predictions for each dataset."""
    model_errors = (
        results.groupby(["dataset", "model"])["absolute_error"]
        .mean()
        .reset_index()
    )
    best_models = model_errors.loc[model_errors.groupby("dataset")["absolute_error"].idxmin()]

    plot_frames = []
    for _, row in best_models.iterrows():
        mask = (results["dataset"] == row["dataset"]) & (results["model"] == row["model"])
        plot_frames.append(results.loc[mask])
    plot_df = pd.concat(plot_frames, ignore_index=True)

    plt.figure(figsize=(9, 6))
    for label, group in plot_df.groupby(["dataset", "model"]):
        dataset_name, model_name = label
        plt.scatter(
            group["actual_power_output"],
            group["predicted_power_output"],
            alpha=0.4,
            s=16,
            label=f"{dataset_name}: {model_name}",
        )

    max_power = max(
        float(plot_df["actual_power_output"].max()),
        float(plot_df["predicted_power_output"].max()),
    )
    plt.plot([0, max_power], [0, max_power], color="black", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual power output (kW)")
    plt.ylabel("Predicted power output (kW)")
    plt.title("Best Model Predictions by Dataset")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=160)
    plt.close()


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> None:
    """Create a grouped comparison chart for RMSE across datasets."""
    pivot = metrics_df.pivot(index="model", columns="dataset", values="rmse")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    ax = pivot.plot(kind="barh", figsize=(11, 7), width=0.8)
    ax.set_title("RMSE Comparison Across Weather Datasets")
    ax.set_xlabel("RMSE (kW)")
    ax.set_ylabel("Model")
    ax.invert_yaxis()
    ax.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=160)
    plt.close()


def train_on_dataset(dataset_name: str, df: pd.DataFrame) -> tuple[list, list, dict]:
    """Train every model on one dataset and return metrics, predictions, and trained models."""
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")
    if len(df) < 50:
        raise ValueError(f"{dataset_name} has too few rows for training: {len(df)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = []
    prediction_frames = []
    trained_models = {}

    for model_name, model in build_models().items():
        print(f"Training {model_name} on {dataset_name}...")
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        metrics.append(evaluate_model(dataset_name, model_name, len(df), y_test, y_pred))
        trained_models[(dataset_name, model_name)] = model

        model_predictions = X_test.copy()
        model_predictions["dataset"] = dataset_name
        model_predictions["model"] = model_name
        model_predictions["actual_power_output"] = y_test.to_numpy()
        model_predictions["predicted_power_output"] = np.round(y_pred, 4)
        model_predictions["absolute_error"] = np.abs(
            model_predictions["actual_power_output"] - model_predictions["predicted_power_output"]
        )
        prediction_frames.append(model_predictions)

    return metrics, prediction_frames, trained_models


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_training_datasets()
    all_metrics = []
    all_prediction_frames = []
    all_trained_models = {}

    for dataset_name, df in datasets.items():
        metrics, prediction_frames, trained_models = train_on_dataset(dataset_name, df)
        all_metrics.extend(metrics)
        all_prediction_frames.extend(prediction_frames)
        all_trained_models.update(trained_models)

    metrics_df = pd.DataFrame(all_metrics).sort_values(["dataset", "rmse"])
    prediction_results = pd.concat(all_prediction_frames, ignore_index=True)
    best_models_df = metrics_df.loc[metrics_df.groupby("dataset")["rmse"].idxmin()].copy()

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    prediction_results.to_csv(PREDICTION_RESULTS_PATH, index=False)
    best_models_df.to_csv(BEST_MODELS_PATH, index=False)
    save_prediction_plot(prediction_results)
    save_model_comparison_plot(metrics_df)

    overall_best = metrics_df.sort_values("rmse").iloc[0]
    best_key = (overall_best["dataset"], overall_best["model"])
    joblib.dump(
        {
            "model": all_trained_models[best_key],
            "dataset": overall_best["dataset"],
            "model_name": overall_best["model"],
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        BEST_MODEL_PATH,
    )

    print("Model metrics by dataset:")
    print(metrics_df.to_string(index=False))
    print(f"Saved model metrics: {MODEL_METRICS_PATH}")
    print(f"Saved best models by dataset: {BEST_MODELS_PATH}")
    print(f"Saved prediction results: {PREDICTION_RESULTS_PATH}")
    print(f"Saved prediction plot: {PREDICTION_PLOT_PATH}")
    print(f"Saved model comparison plot: {MODEL_COMPARISON_PLOT_PATH}")
    print(f"Saved overall best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
