"""Compare environment-only and electrical-assisted PV power models."""

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

QUALITY_PATH = RESULTS_DIR / "real_data_feature_quality_report.csv"
CORRELATION_PATH = RESULTS_DIR / "real_data_correlation_report.csv"
CORRELATION_PLOT_PATH = RESULTS_DIR / "real_data_correlation_heatmap.png"
COMPARISON_PATH = RESULTS_DIR / "environment_vs_electrical_model_comparison.csv"
COMPARISON_PLOT_PATH = RESULTS_DIR / "environment_vs_electrical_model_comparison.png"

MODEL_DESIGNS = {
    "Environment-based model": ["irradiance", "temperature"],
    "Electrical-assisted model": ["irradiance", "temperature", "voltage", "current"],
}
IMPORTANCE_PATHS = {
    "Environment-based model": (
        RESULTS_DIR / "feature_importance_environment_model.csv",
        RESULTS_DIR / "feature_importance_environment_model.png",
    ),
    "Electrical-assisted model": (
        RESULTS_DIR / "feature_importance_electrical_model.csv",
        RESULTS_DIR / "feature_importance_electrical_model.png",
    ),
}
BASELINE_PATHS = {
    "Environment-based model": RESULTS_DIR / "predicted_power_environment_baseline.csv",
    "Electrical-assisted model": RESULTS_DIR / "predicted_power_electrical_assisted_baseline.csv",
}
REPORT_FEATURES = ["current", "voltage", "irradiance", "temperature", "power"]
TARGET = "power"


def save_quality_report(df: pd.DataFrame) -> None:
    """Save basic distribution and missing-value evidence for model fields."""
    rows = []
    for feature in REPORT_FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    pd.DataFrame(rows).to_csv(QUALITY_PATH, index=False)


def save_correlation_report(df: pd.DataFrame) -> None:
    """Save correlations with power and a full correlation heatmap."""
    correlation = df[REPORT_FEATURES].corr()
    report = (
        correlation[TARGET]
        .drop(TARGET)
        .rename("correlation_with_power")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("correlation_with_power", ascending=False)
    )
    report.to_csv(CORRELATION_PATH, index=False)

    figure, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation.iloc[row, column]) > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Real PVDAQ Feature Correlation Matrix")
    figure.tight_layout()
    figure.savefig(CORRELATION_PLOT_PATH, dpi=160)
    plt.close(figure)


def save_feature_importance(model_name: str, model, features: list[str]) -> None:
    """Fit one model design and save its tree-based feature importance."""
    values = np.asarray(model.feature_importances_, dtype=float)
    result = pd.DataFrame(
        {
            "model_design": model_name,
            "feature": features,
            "importance": values,
            "importance_percent": values / values.sum() * 100,
            "analysis_method": "feature_importances_",
        }
    ).sort_values("importance", ascending=False)
    csv_path, plot_path = IMPORTANCE_PATHS[model_name]
    result.to_csv(csv_path, index=False)

    plot_data = result.sort_values("importance")
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(plot_data["feature"], plot_data["importance_percent"], color="#4c78a8")
    for bar, value in zip(bars, plot_data["importance_percent"]):
        axis.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center")
    axis.set_xlim(0, float(plot_data["importance_percent"].max()) + 12)
    axis.set_xlabel("Feature importance (%)")
    axis.set_title(f"Feature Importance: {model_name}")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def save_model_comparison(results: pd.DataFrame) -> None:
    """Plot comparable out-of-fold MAE, RMSE, and R-squared values."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#59a14f", "#4c78a8"]
    labels = ["Environment", "Electrical-assisted"]
    for axis, metric, title in zip(
        axes,
        ["mae", "rmse", "r2"],
        ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"],
    ):
        bars = axis.bar(labels, results[metric], color=colors)
        for bar, value in zip(bars, results[metric]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Environment-based vs Electrical-assisted Random Forest")
    figure.tight_layout()
    figure.savefig(COMPARISON_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run quality checks and compare the two feature-input designs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    save_quality_report(df)
    save_correlation_report(df)

    base_model = joblib.load(MODEL_PATH)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for model_name, features in MODEL_DESIGNS.items():
        clean = df.dropna(subset=features + [TARGET]).copy()
        prediction = cross_val_predict(
            clone(base_model), clean[features], clean[TARGET], cv=cv, n_jobs=-1
        )
        baseline = pd.DataFrame(
            {
                "time": clean["time"],
                "source_site": clean["source_site"],
                "actual_power": clean[TARGET],
                "predicted_power": prediction,
                "prediction_error": clean[TARGET].to_numpy() - prediction,
                "absolute_error": np.abs(clean[TARGET].to_numpy() - prediction),
            }
        )
        baseline.to_csv(BASELINE_PATHS[model_name], index=False)
        fitted_model = clone(base_model).fit(clean[features], clean[TARGET])
        save_feature_importance(model_name, fitted_model, features)
        rows.append(
            {
                "model_design": model_name,
                "features": ", ".join(features),
                "rows": len(clean),
                "validation": "5-fold shuffled out-of-fold prediction",
                "mae": mean_absolute_error(clean[TARGET], prediction),
                "rmse": mean_squared_error(clean[TARGET], prediction) ** 0.5,
                "r2": r2_score(clean[TARGET], prediction),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(COMPARISON_PATH, index=False)
    save_model_comparison(results)

    print("Feature quality report:")
    print(pd.read_csv(QUALITY_PATH).to_string(index=False))
    print("\nCorrelation with power:")
    print(pd.read_csv(CORRELATION_PATH).to_string(index=False))
    print("\nModel input design comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
