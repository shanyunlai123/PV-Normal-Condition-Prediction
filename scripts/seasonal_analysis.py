"""Analyze seasonal PV conditions and cross-site prediction difficulty."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from research_validation_utils import FEATURES, TARGET, build_model, load_validation_data


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "seasonal_statistics.csv"
PLOT_PATH = RESULTS_DIR / "seasonal_comparison.png"
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


def main() -> None:
    """Save seasonal descriptive statistics and group-held-out model metrics."""
    df = load_validation_data()
    groups = df["system_id"]
    cv = GroupKFold(n_splits=groups.nunique())
    df["predicted_normalized_power"] = cross_val_predict(
        build_model(), df[FEATURES], df[TARGET], groups=groups, cv=cv, n_jobs=-1
    )

    rows = []
    for season in SEASON_ORDER:
        group = df[df["season"] == season]
        rows.append(
            {
                "season": season,
                "rows": len(group),
                "sites": group["system_id"].nunique(),
                "irradiance_mean": group["irradiance"].mean(),
                "irradiance_std": group["irradiance"].std(),
                "temperature_mean": group["temperature"].mean(),
                "temperature_std": group["temperature"].std(),
                "power_kw_mean": group["power_kw"].mean(),
                "power_kw_std": group["power_kw"].std(),
                "normalized_power_mean": group[TARGET].mean(),
                "cross_site_mae": mean_absolute_error(
                    group[TARGET], group["predicted_normalized_power"]
                ),
                "cross_site_rmse": mean_squared_error(
                    group[TARGET], group["predicted_normalized_power"]
                ) ** 0.5,
                "cross_site_r2": r2_score(
                    group[TARGET], group["predicted_normalized_power"]
                ),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    results.plot.bar(x="season", y="irradiance_mean", ax=axes[0], legend=False, color="#e6a700")
    results.plot.bar(x="season", y="temperature_mean", ax=axes[1], legend=False, color="#d95f59")
    results.plot.bar(x="season", y="cross_site_rmse", ax=axes[2], legend=False, color="#4c78a8")
    axes[0].set_title("Mean Daytime Irradiance")
    axes[1].set_title("Mean Temperature")
    axes[2].set_title("Cross-Site Normalized RMSE")
    for axis in axes:
        axis.set_xlabel("Season")
        axis.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    print(results.to_string(index=False))
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
