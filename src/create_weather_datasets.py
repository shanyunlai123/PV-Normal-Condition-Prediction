from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
WEATHER_DATA_DIR = DATA_DIR / "weather_datasets"

CLEAN_DATA_PATH = DATA_DIR / "clean_dataset.csv"
ALL_WEATHER_PATH = WEATHER_DATA_DIR / "all_weather_dataset.csv"
SUNNY_PATH = WEATHER_DATA_DIR / "sunny_dataset.csv"
CLOUDY_PATH = WEATHER_DATA_DIR / "cloudy_dataset.csv"
SUMMARY_PATH = RESULTS_DIR / "weather_dataset_summary.csv"
PLOT_PATH = RESULTS_DIR / "weather_dataset_distribution.png"


def load_clean_dataset() -> pd.DataFrame:
    """Read the cleaned PV dataset created by the preprocessing stage."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {CLEAN_DATA_PATH}. Run `python src/preprocess_data.py` first."
        )

    return pd.read_csv(CLEAN_DATA_PATH)


def add_weather_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Label daytime records as sunny, cloudy, or moderate using hourly irradiance quantiles."""
    labeled_df = df.copy()
    labeled_df["weather_condition"] = "night_or_low_power"

    # Only daytime samples are useful for separating sunny and cloudy generation behavior.
    daytime_mask = (
        labeled_df["hour"].between(7, 17)
        & (labeled_df["irradiance"] >= 20)
        & (labeled_df["power_output"] > 0)
    )
    daytime_df = labeled_df.loc[daytime_mask].copy()

    # Compare each sample against records from the same hour, so morning/evening are treated fairly.
    hourly_quantiles = daytime_df.groupby("hour")["irradiance"].quantile([0.33, 0.67]).unstack()
    hourly_quantiles.columns = ["cloudy_threshold", "sunny_threshold"]
    daytime_df = daytime_df.join(hourly_quantiles, on="hour")

    sunny_index = daytime_df.index[
        daytime_df["irradiance"] >= daytime_df["sunny_threshold"]
    ]
    cloudy_index = daytime_df.index[
        daytime_df["irradiance"] <= daytime_df["cloudy_threshold"]
    ]
    moderate_index = daytime_df.index.difference(sunny_index.union(cloudy_index))

    labeled_df.loc[sunny_index, "weather_condition"] = "sunny"
    labeled_df.loc[cloudy_index, "weather_condition"] = "cloudy"
    labeled_df.loc[moderate_index, "weather_condition"] = "moderate"

    return labeled_df


def save_summary(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Save row counts and basic power statistics for each weather dataset."""
    summary = (
        labeled_df.groupby("weather_condition")
        .agg(
            rows=("power_output", "size"),
            avg_irradiance=("irradiance", "mean"),
            avg_power_output=("power_output", "mean"),
            max_power_output=("power_output", "max"),
        )
        .reset_index()
        .round(3)
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def save_distribution_plot(labeled_df: pd.DataFrame) -> None:
    """Visualize how sunny and cloudy datasets differ in irradiance and power output."""
    plot_df = labeled_df[labeled_df["weather_condition"].isin(["sunny", "cloudy", "moderate"])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition, group in plot_df.groupby("weather_condition"):
        axes[0].hist(group["irradiance"], bins=35, alpha=0.55, label=condition)
        axes[1].hist(group["power_output"], bins=35, alpha=0.55, label=condition)

    axes[0].set_title("Irradiance by Weather Dataset")
    axes[0].set_xlabel("Irradiance")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].set_title("Power Output by Weather Dataset")
    axes[1].set_xlabel("Power output (kW)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    WEATHER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = load_clean_dataset()
    labeled_df = add_weather_condition(clean_df)

    sunny_df = labeled_df[labeled_df["weather_condition"] == "sunny"].copy()
    cloudy_df = labeled_df[labeled_df["weather_condition"] == "cloudy"].copy()

    labeled_df.to_csv(ALL_WEATHER_PATH, index=False)
    sunny_df.to_csv(SUNNY_PATH, index=False)
    cloudy_df.to_csv(CLOUDY_PATH, index=False)

    summary = save_summary(labeled_df)
    save_distribution_plot(labeled_df)

    print("Weather dataset summary:")
    print(summary.to_string(index=False))
    print(f"Saved all-weather dataset: {ALL_WEATHER_PATH}")
    print(f"Saved sunny dataset: {SUNNY_PATH}")
    print(f"Saved cloudy dataset: {CLOUDY_PATH}")
    print(f"Saved weather dataset summary: {SUMMARY_PATH}")
    print(f"Saved weather dataset plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
