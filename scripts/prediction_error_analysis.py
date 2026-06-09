"""Generate honest out-of-fold prediction errors and diagnostic plots."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "all_weather_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_random_forest.pkl"
RESULTS_DIR = ROOT_DIR / "results"

BASELINE_PATH = RESULTS_DIR / "predicted_power_baseline.csv"
ACTUAL_PREDICTED_PATH = RESULTS_DIR / "actual_vs_predicted.png"
RESIDUAL_PATH = RESULTS_DIR / "residual_plot.png"
ERROR_DISTRIBUTION_PATH = RESULTS_DIR / "error_distribution.png"
SUMMARY_PATH = RESULTS_DIR / "prediction_error_summary.csv"

FEATURES = ["irradiance", "temperature", "voltage", "current"]
TARGET = "power"


def save_actual_vs_predicted(actual: pd.Series, predicted: np.ndarray) -> None:
    """Plot actual power against out-of-fold predicted power."""
    limit = max(float(actual.max()), float(predicted.max()))
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.3, s=12)
    plt.plot([0, limit], [0, limit], color="black", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual power")
    plt.ylabel("Predicted power")
    plt.title("Actual vs Predicted Power (Out-of-Fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_PREDICTED_PATH, dpi=160)
    plt.close()


def save_residual_plot(predicted: np.ndarray, residual: np.ndarray) -> None:
    """Plot residuals against predicted power to inspect systematic error."""
    plt.figure(figsize=(8, 5.5))
    plt.scatter(predicted, residual, alpha=0.3, s=12)
    plt.axhline(0, color="black", linewidth=2)
    plt.xlabel("Predicted power")
    plt.ylabel("Residual: actual - predicted")
    plt.title("Residual Plot (Out-of-Fold)")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PATH, dpi=160)
    plt.close()


def save_error_distribution(residual: np.ndarray) -> None:
    """Plot the residual distribution."""
    plt.figure(figsize=(8, 5.5))
    plt.hist(residual, bins=50, color="#4c78a8", alpha=0.85)
    plt.axvline(0, color="black", linewidth=2)
    plt.xlabel("Prediction residual")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution (Out-of-Fold)")
    plt.tight_layout()
    plt.savefig(ERROR_DISTRIBUTION_PATH, dpi=160)
    plt.close()


def main() -> None:
    """Create out-of-fold predictions, baseline data, plots, and error evidence."""
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    clean = df.dropna(subset=FEATURES + [TARGET]).copy().reset_index(drop=True)
    model = joblib.load(MODEL_PATH)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Out-of-fold predictions ensure each row is predicted by a model that did not train on it.
    predicted = cross_val_predict(clone(model), clean[FEATURES], clean[TARGET], cv=cv, n_jobs=-1)
    predicted = np.clip(predicted, 0, None)
    residual = clean[TARGET].to_numpy() - predicted

    baseline = pd.DataFrame(
        {
            "time": clean["time"],
            "actual_power": clean[TARGET],
            "predicted_power": predicted,
            "prediction_error": residual,
            "absolute_error": np.abs(residual),
            "source_site": clean["source_site"],
            "weather_condition": clean["weather_condition"],
        }
    )
    baseline.to_csv(BASELINE_PATH, index=False)

    condition_rows = []
    for condition, group in baseline.groupby("weather_condition"):
        condition_rows.append(
            {
                "weather_condition": condition,
                "rows": len(group),
                "mean_error": group["prediction_error"].mean(),
                "mae": group["absolute_error"].mean(),
                "error_std": group["prediction_error"].std(),
                "rmse": np.sqrt(np.mean(group["prediction_error"] ** 2)),
                "r2": r2_score(group["actual_power"], group["predicted_power"])
                if group["actual_power"].nunique() > 1
                else np.nan,
            }
        )
    error_by_condition = pd.DataFrame(condition_rows)
    overall = pd.DataFrame(
        [
            {
                "weather_condition": "overall",
                "rows": len(baseline),
                "mean_error": residual.mean(),
                "mae": mean_absolute_error(clean[TARGET], predicted),
                "error_std": residual.std(),
                "rmse": np.sqrt(mean_squared_error(clean[TARGET], predicted)),
                "r2": r2_score(clean[TARGET], predicted),
            }
        ]
    )
    summary = pd.concat([overall, error_by_condition], ignore_index=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    save_actual_vs_predicted(clean[TARGET], predicted)
    save_residual_plot(predicted, residual)
    save_error_distribution(residual)

    print("Prediction error summary:")
    print(summary.to_string(index=False))
    print(f"Saved Module 2 baseline: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
