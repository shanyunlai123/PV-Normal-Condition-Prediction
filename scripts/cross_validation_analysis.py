"""Run K-Fold cross validation for major models across operating scenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_CSV_PATH = RESULTS_DIR / "cross_validation_results.csv"
OUTPUT_PLOT_PATH = RESULTS_DIR / "cross_validation_comparison.png"

SCENARIOS = [
    "all_weather",
    "sunny",
    "cloudy",
    "moderate",
    "rainy",
    "high_temperature",
    "low_temperature",
]
FEATURES = ["irradiance", "temperature", "voltage", "current"]
TARGET = "power"


def build_models() -> dict:
    """Return the four required existing model families."""
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Lasso Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", Lasso(alpha=0.001, max_iter=10000))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def prepare_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one scenario and keep rows with complete model inputs."""
    df = pd.read_csv(path)
    available_features = [feature for feature in FEATURES if feature in df.columns]
    clean = df.dropna(subset=available_features + [TARGET]).copy()
    return clean[available_features], clean[TARGET]


def main() -> None:
    """Run five-fold CV and save mean and standard-deviation metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for scenario in SCENARIOS:
        path = PROCESSED_DIR / f"{scenario}_dataset.csv"
        if not path.exists():
            print(f"Skipping missing scenario: {scenario}")
            continue

        X, y = prepare_dataset(path)
        if len(X) < 25:
            print(f"Skipping {scenario}: only {len(X)} complete rows")
            continue

        n_splits = min(5, max(2, len(X) // 20))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model_name, model in build_models().items():
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            rows.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "rows": len(X),
                    "folds": n_splits,
                    "mae_mean": -scores["test_mae"].mean(),
                    "mae_std": scores["test_mae"].std(),
                    "rmse_mean": -scores["test_rmse"].mean(),
                    "rmse_std": scores["test_rmse"].std(),
                    "r2_mean": scores["test_r2"].mean(),
                    "r2_std": scores["test_r2"].std(),
                }
            )

    results = pd.DataFrame(rows).sort_values(["scenario", "rmse_mean"])
    results.to_csv(OUTPUT_CSV_PATH, index=False)

    pivot = results.pivot(index="model", columns="scenario", values="rmse_mean")
    ax = pivot.plot(kind="barh", figsize=(12, 8))
    ax.set_title("Cross Validation RMSE by Model and Operating Scenario")
    ax.set_xlabel("Mean CV RMSE")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close()

    print("Cross validation results:")
    print(results.to_string(index=False))
    print(f"Saved CV results: {OUTPUT_CSV_PATH}")
    print(f"Saved CV comparison plot: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
