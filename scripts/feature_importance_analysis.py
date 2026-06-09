"""Compare feature importance for the best model on each weather dataset."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data" / "weather_datasets"

# Reuse the existing model definitions so this script does not add new models.
sys.path.insert(0, str(SRC_DIR))
from train_model import FEATURE_COLUMNS, TARGET_COLUMN, build_models  # noqa: E402


METRICS_PATH = RESULTS_DIR / "model_metrics.csv"
COMPARISON_PLOT_PATH = RESULTS_DIR / "feature_importance_comparison.png"

DATASET_PATHS = {
    "all_weather": DATA_DIR / "all_weather_dataset.csv",
    "sunny": DATA_DIR / "sunny_dataset.csv",
    "cloudy": DATA_DIR / "cloudy_dataset.csv",
}

OUTPUT_PATHS = {
    "all_weather": RESULTS_DIR / "feature_importance_all_weather.csv",
    "sunny": RESULTS_DIR / "feature_importance_sunny.csv",
    "cloudy": RESULTS_DIR / "feature_importance_cloudy.csv",
}

# Temperature combines ambient and module temperature. Voltage and current are
# included in the report, but the current training datasets do not contain them.
FEATURE_GROUPS = {
    "irradiance": ["irradiance"],
    "temperature": ["ambient_temperature", "module_temperature"],
    "voltage": ["voltage", "ac_voltage", "dc_voltage"],
    "current": ["current", "ac_current", "dc_current"],
}


def load_best_model_names() -> dict[str, str]:
    """Read existing metrics and select the lowest-RMSE model per dataset."""
    metrics = pd.read_csv(METRICS_PATH)
    best_rows = metrics.loc[metrics.groupby("dataset")["rmse"].idxmin()]
    return dict(zip(best_rows["dataset"], best_rows["model"]))


def unwrap_estimator(model):
    """Return the final estimator when an existing model uses a Pipeline."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def extract_importance(model, feature_names: list[str]) -> tuple[np.ndarray, str]:
    """Extract tree importance or standardized absolute linear coefficients."""
    estimator = unwrap_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float), "feature_importances_"

    if hasattr(estimator, "coef_"):
        # Linear models are defined in a StandardScaler pipeline, so these are
        # standardized coefficient magnitudes and can be compared directly.
        return np.abs(np.ravel(np.asarray(estimator.coef_, dtype=float))), "standardized_abs_coef"

    raise ValueError(f"Model {type(estimator).__name__} does not expose supported importance values.")


def aggregate_requested_features(
    feature_names: list[str],
    raw_values: np.ndarray,
    dataset: str,
    model_name: str,
    method: str,
) -> pd.DataFrame:
    """Aggregate detailed model inputs into the four requested feature groups."""
    raw_df = pd.DataFrame({"source_feature": feature_names, "raw_importance": raw_values})
    rows = []

    for feature_group, source_names in FEATURE_GROUPS.items():
        matched = raw_df[raw_df["source_feature"].isin(source_names)]
        rows.append(
            {
                "dataset": dataset,
                "best_model": model_name,
                "feature": feature_group,
                "importance": float(matched["raw_importance"].sum()),
                "used_in_model": not matched.empty,
                "source_features": ", ".join(matched["source_feature"]) or "not used",
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


def analyze_dataset(dataset: str, model_name: str) -> pd.DataFrame:
    """Fit the existing best model and calculate its feature importance."""
    dataset_df = pd.read_csv(DATASET_PATHS[dataset])
    X = dataset_df[FEATURE_COLUMNS]
    y = dataset_df[TARGET_COLUMN]

    existing_models = build_models()
    model = existing_models[model_name]
    model.fit(X, y)

    raw_values, method = extract_importance(model, FEATURE_COLUMNS)
    result = aggregate_requested_features(
        FEATURE_COLUMNS,
        raw_values,
        dataset,
        model_name,
        method,
    )
    result.to_csv(OUTPUT_PATHS[dataset], index=False)
    return result


def save_comparison_plot(all_results: pd.DataFrame) -> None:
    """Create a grouped matplotlib chart comparing feature importance."""
    pivot = all_results.pivot(index="feature", columns="dataset", values="importance_percent")
    pivot = pivot.reindex(["irradiance", "temperature", "voltage", "current"])

    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_title("Feature Importance by Best Model and Weather Dataset")
    ax.set_xlabel("Requested feature")
    ax.set_ylabel("Normalized importance (%)")
    ax.legend(title="Dataset")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    """Run feature importance evidence analysis for all three datasets."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_names = load_best_model_names()
    result_frames = []

    for dataset in DATASET_PATHS:
        model_name = best_model_names[dataset]
        print(f"Analyzing {dataset} best model: {model_name}")
        result_frames.append(analyze_dataset(dataset, model_name))

    all_results = pd.concat(result_frames, ignore_index=True)
    save_comparison_plot(all_results)

    print("\nMost important used feature by dataset:")
    for dataset, group in all_results[all_results["used_in_model"]].groupby("dataset"):
        top = group.sort_values("importance", ascending=False).iloc[0]
        print(f"- {dataset}: {top['feature']} ({top['importance_percent']:.2f}%)")

    print(f"Saved comparison plot: {COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    main()
