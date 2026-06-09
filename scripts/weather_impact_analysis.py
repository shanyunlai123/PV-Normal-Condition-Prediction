"""Calculate statistical evidence for weather impacts on PV prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "weather_datasets"
RESULTS_DIR = ROOT_DIR / "results"

OUTPUT_CSV_PATH = RESULTS_DIR / "weather_statistics.csv"
OUTPUT_PLOT_PATH = RESULTS_DIR / "weather_statistics.png"

DATASET_PATHS = {
    "all_weather": DATA_DIR / "all_weather_dataset.csv",
    "sunny": DATA_DIR / "sunny_dataset.csv",
    "cloudy": DATA_DIR / "cloudy_dataset.csv",
}


def calculate_statistics(dataset_name: str, dataset_path: Path) -> dict:
    """Calculate required weather and power statistics for one dataset."""
    df = pd.read_csv(dataset_path)
    irradiance_mean = float(df["irradiance"].mean())
    power_mean = float(df["power_output"].mean())

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "irradiance_mean": irradiance_mean,
        "irradiance_std": float(df["irradiance"].std()),
        "irradiance_cv": float(df["irradiance"].std() / irradiance_mean) if irradiance_mean else np.nan,
        "power_mean": power_mean,
        "power_std": float(df["power_output"].std()),
        "power_cv": float(df["power_output"].std() / power_mean) if power_mean else np.nan,
        "irradiance_power_correlation": float(df["irradiance"].corr(df["power_output"])),
    }


def save_statistics_plot(statistics_df: pd.DataFrame) -> None:
    """Create matplotlib charts that provide evidence for weather differences."""
    plot_df = statistics_df.set_index("dataset").loc[["all_weather", "sunny", "cloudy"]]
    colors = ["#4c78a8", "#f2cf5b", "#7a9cc6"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    plot_df["irradiance_mean"].plot(
        kind="bar", ax=axes[0, 0], color=colors, title="Mean Irradiance"
    )
    axes[0, 0].set_ylabel("Irradiance")

    plot_df["irradiance_std"].plot(
        kind="bar", ax=axes[0, 1], color=colors, title="Irradiance Standard Deviation"
    )
    axes[0, 1].set_ylabel("Irradiance standard deviation")

    plot_df[["power_mean", "power_std"]].plot(
        kind="bar", ax=axes[1, 0], title="Power Mean and Standard Deviation"
    )
    axes[1, 0].set_ylabel("Power output (kW)")

    plot_df["irradiance_power_correlation"].plot(
        kind="bar", ax=axes[1, 1], color=colors, title="Irradiance-Power Correlation"
    )
    axes[1, 1].set_ylabel("Correlation")
    axes[1, 1].set_ylim(0, 1.02)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Weather Impact Evidence")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close()


def print_evidence(statistics_df: pd.DataFrame) -> None:
    """Print evidence without making claims that are not supported by the data."""
    indexed = statistics_df.set_index("dataset")
    sunny = indexed.loc["sunny"]
    cloudy = indexed.loc["cloudy"]

    print("\nEvidence-based comparison:")
    print(
        f"- Sunny mean irradiance is {sunny['irradiance_mean']:.3f}, "
        f"compared with {cloudy['irradiance_mean']:.3f} for cloudy."
    )
    print(
        f"- Sunny irradiance standard deviation is {sunny['irradiance_std']:.3f}; "
        f"cloudy irradiance standard deviation is {cloudy['irradiance_std']:.3f}."
    )
    print(
        f"- Relative irradiance variability (CV) is {sunny['irradiance_cv']:.3f} for sunny "
        f"and {cloudy['irradiance_cv']:.3f} for cloudy."
    )
    print(
        f"- Irradiance-power correlation is {sunny['irradiance_power_correlation']:.4f} "
        f"for sunny and {cloudy['irradiance_power_correlation']:.4f} for cloudy."
    )


def main() -> None:
    """Generate the weather statistics evidence table and chart."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        calculate_statistics(dataset_name, dataset_path)
        for dataset_name, dataset_path in DATASET_PATHS.items()
    ]
    statistics_df = pd.DataFrame(rows).round(6)
    statistics_df.to_csv(OUTPUT_CSV_PATH, index=False)
    save_statistics_plot(statistics_df)

    print("Weather statistics:")
    print(statistics_df.to_string(index=False))
    print_evidence(statistics_df)
    print(f"Saved weather statistics CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved weather statistics plot: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
