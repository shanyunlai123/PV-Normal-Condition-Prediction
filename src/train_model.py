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
CLEAN_DATA_PATH = ROOT_DIR / "data" / "clean_dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

PREDICTION_RESULTS_PATH = RESULTS_DIR / "prediction_results.csv"
PREDICTION_PLOT_PATH = RESULTS_DIR / "predicted_vs_actual.png"
MODEL_COMPARISON_PLOT_PATH = RESULTS_DIR / "model_comparison.png"
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
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            random_state=42,
        ),
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


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_prediction_plot(results: pd.DataFrame) -> None:
    """Plot predicted values against actual values for every model."""
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


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> None:
    """Create a compact chart to compare model errors and R2 scores."""
    sorted_metrics = metrics_df.sort_values("rmse", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(sorted_metrics["model"], sorted_metrics["mae"], color="#4c78a8")
    axes[0].set_title("MAE by Model")
    axes[0].set_xlabel("MAE (kW)")
    axes[0].invert_yaxis()

    axes[1].barh(sorted_metrics["model"], sorted_metrics["rmse"], color="#f58518")
    axes[1].set_title("RMSE by Model")
    axes[1].set_xlabel("RMSE (kW)")
    axes[1].invert_yaxis()

    axes[2].barh(sorted_metrics["model"], sorted_metrics["r2"], color="#54a24b")
    axes[2].set_title("R2 by Model")
    axes[2].set_xlabel("R2 score")
    axes[2].invert_yaxis()

    fig.suptitle("PV Power Prediction Model Comparison")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=160)
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
        print(f"Training {model_name}...")
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
    save_model_comparison_plot(metrics_df)

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
    print(f"Saved model comparison plot: {MODEL_COMPARISON_PLOT_PATH}")
    print(f"Saved model metrics: {MODEL_METRICS_PATH}")
    print(f"Saved best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
