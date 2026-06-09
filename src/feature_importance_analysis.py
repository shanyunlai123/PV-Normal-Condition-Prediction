from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_pv_power_model.joblib"
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "feature_importance.csv"
PLOT_PATH = RESULTS_DIR / "feature_importance.png"

# Group the model's detailed features into the four variables requested for analysis.
FEATURE_GROUPS = {
    "irradiance": ["irradiance"],
    "temperature": ["ambient_temperature", "module_temperature"],
    "voltage": ["voltage", "ac_voltage", "dc_voltage"],
    "current": ["current", "ac_current", "dc_current"],
}

TREE_MODEL_NAMES = {"Random Forest", "Gradient Boosting", "Extra Trees", "Decision Tree"}
LINEAR_MODEL_NAMES = {"Linear Regression", "Lasso Regression", "Ridge Regression"}


def load_best_model() -> dict:
    """Load the best trained model and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing best model: {MODEL_PATH}. Run `python src/train_model.py` first."
        )

    return joblib.load(MODEL_PATH)


def unwrap_model(model):
    """Return the final estimator when the saved model is a sklearn Pipeline."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def extract_raw_importance(bundle: dict) -> tuple[pd.DataFrame, str]:
    """Extract feature importance from a tree model or coefficients from a linear model."""
    model_name = bundle["model_name"]
    feature_columns = bundle["feature_columns"]
    estimator = unwrap_model(bundle["model"])

    if model_name in TREE_MODEL_NAMES and hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif model_name in LINEAR_MODEL_NAMES and hasattr(estimator, "coef_"):
        # Absolute coefficients represent influence magnitude; coefficient sign is kept separately.
        coefficients = np.ravel(np.asarray(estimator.coef_, dtype=float))
        values = np.abs(coefficients)
        method = "absolute_model_coefficients"
    else:
        raise ValueError(
            f"Feature importance analysis is not supported for best model: {model_name}"
        )

    if len(values) != len(feature_columns):
        raise ValueError("The number of model importance values does not match feature columns.")

    raw_df = pd.DataFrame(
        {
            "source_feature": feature_columns,
            "raw_importance": values,
        }
    )
    return raw_df, method


def aggregate_requested_variables(raw_df: pd.DataFrame, bundle: dict, method: str) -> pd.DataFrame:
    """Aggregate detailed model features into irradiance, temperature, voltage, and current."""
    rows = []

    for variable, possible_features in FEATURE_GROUPS.items():
        matched = raw_df[raw_df["source_feature"].isin(possible_features)]
        rows.append(
            {
                "variable": variable,
                "importance": float(matched["raw_importance"].sum()),
                "used_in_model": not matched.empty,
                "source_features": ", ".join(matched["source_feature"].tolist()) or "not used",
                "model_name": bundle["model_name"],
                "dataset": bundle.get("dataset", "unknown"),
                "analysis_method": method,
            }
        )

    result = pd.DataFrame(rows)
    used_total = result.loc[result["used_in_model"], "importance"].sum()
    result["importance_percent"] = np.where(
        result["used_in_model"] & (used_total > 0),
        result["importance"] / used_total * 100,
        0.0,
    )

    return result.sort_values(
        ["used_in_model", "importance"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_plot(result: pd.DataFrame, bundle: dict) -> None:
    """Create a matplotlib feature importance chart."""
    plot_df = result.sort_values("importance", ascending=True)
    colors = ["#4c78a8" if used else "#b8b8b8" for used in plot_df["used_in_model"]]

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(plot_df["variable"], plot_df["importance_percent"], color=colors)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        if row["used_in_model"]:
            label = f'{row["importance_percent"]:.1f}%'
        else:
            label = "not used in model"
        plt.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    plt.xlabel("Normalized importance among requested variables (%)")
    plt.ylabel("Variable")
    plt.title(
        f'Feature Importance Analysis\nBest model: {bundle["model_name"]} '
        f'({bundle.get("dataset", "unknown")})'
    )
    plt.xlim(0, max(105, float(plot_df["importance_percent"].max()) + 15))
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_best_model()
    raw_df, method = extract_raw_importance(bundle)
    result = aggregate_requested_variables(raw_df, bundle, method)

    result.to_csv(CSV_PATH, index=False)
    save_plot(result, bundle)

    used_result = result[result["used_in_model"]]
    most_important = used_result.iloc[0]

    print("Feature importance analysis:")
    print(result.to_string(index=False))
    print(
        f'Most influential requested variable: {most_important["variable"]} '
        f'({most_important["importance_percent"]:.2f}%)'
    )
    print(f"Saved feature importance CSV: {CSV_PATH}")
    print(f"Saved feature importance plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
