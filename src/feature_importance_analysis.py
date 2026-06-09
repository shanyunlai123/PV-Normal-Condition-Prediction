"""Analyze feature importance for the best tuned real-data model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_random_forest.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "all_weather_dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "feature_importance.csv"
PLOT_PATH = RESULTS_DIR / "feature_importance.png"

FEATURES = ["irradiance", "temperature", "voltage", "current"]
TARGET = "power"


def main() -> None:
    """Extract and plot feature importance from the tuned Random Forest model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing tuned model: {MODEL_PATH}. Run `python scripts/hyperparameter_tuning.py` first."
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed data: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + [TARGET])

    # Refit the already selected and tuned model on the complete normal dataset.
    model.fit(df[FEATURES], df[TARGET])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(np.asarray(model.coef_, dtype=float)))
        method = "absolute_coefficients"
    else:
        raise ValueError("The selected model does not expose supported importance values.")

    result = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": values,
            "importance_percent": values / values.sum() * 100,
            "model": type(model).__name__,
            "analysis_method": method,
        }
    ).sort_values("importance", ascending=False)
    result.to_csv(CSV_PATH, index=False)

    plot_df = result.sort_values("importance")
    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(plot_df["feature"], plot_df["importance_percent"], color="#4c78a8")
    for bar, value in zip(bars, plot_df["importance_percent"]):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )
    plt.xlabel("Feature importance (%)")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Tuned Real-Data Random Forest")
    plt.xlim(0, float(plot_df["importance_percent"].max()) + 12)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    print("Feature importance:")
    print(result.to_string(index=False))
    print(f"Most important feature: {result.iloc[0]['feature']}")
    print(f"Saved feature importance CSV: {CSV_PATH}")
    print(f"Saved feature importance plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
