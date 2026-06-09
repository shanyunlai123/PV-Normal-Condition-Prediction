"""Tune Random Forest and Gradient Boosting using existing real PVDAQ data."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "all_weather_dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_PATH = RESULTS_DIR / "hyperparameter_tuning_results.csv"

FEATURES = ["irradiance", "temperature", "voltage", "current"]
TARGET = "power"


def evaluate_baseline(model, X: pd.DataFrame, y: pd.Series, cv: KFold) -> float:
    """Return baseline mean cross-validation RMSE."""
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    return float(-scores.mean())


def main() -> None:
    """Tune two ensemble models and save the best estimators."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    searches = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [8, 14, None],
                "min_samples_split": [2, 5],
            },
            "output": MODELS_DIR / "best_random_forest.pkl",
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.08],
                "max_depth": [2, 4],
            },
            "output": MODELS_DIR / "best_gradient_boosting.pkl",
        },
    }

    rows = []
    for model_name, config in searches.items():
        print(f"Tuning {model_name}...")
        baseline_rmse = evaluate_baseline(config["model"], X, y, cv)
        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        best_rmse = float(-search.best_score_)
        joblib.dump(search.best_estimator_, config["output"])

        rows.append(
            {
                "model": model_name,
                "rows": len(df),
                "baseline_cv_rmse": baseline_rmse,
                "best_cv_rmse": best_rmse,
                "rmse_improvement": baseline_rmse - best_rmse,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "saved_model": str(config["output"].relative_to(ROOT_DIR)),
            }
        )

    results = pd.DataFrame(rows).sort_values("best_cv_rmse")
    results.to_csv(OUTPUT_PATH, index=False)
    print("Hyperparameter tuning results:")
    print(results.to_string(index=False))
    print(f"Saved tuning results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
