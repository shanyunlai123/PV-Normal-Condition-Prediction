from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

RAW_DATA_PATH = DATA_DIR / "simulated_pv_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_dataset.csv"
MISSING_VALUE_REPORT_PATH = RESULTS_DIR / "missing_values.csv"
DISTRIBUTION_PLOT_PATH = RESULTS_DIR / "data_distribution.png"

REQUIRED_COLUMNS = [
    "irradiance",
    "ambient_temperature",
    "module_temperature",
    "humidity",
    "wind_speed",
    "hour",
    "day_of_year",
    "power_output",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the raw PV CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {path}. Run `python src/generate_data.py` first."
        )

    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    """Check whether all required PV columns exist before preprocessing."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def save_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values and save a simple CSV report."""
    missing_report = (
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing_report["missing_percent"] = (
        missing_report["missing_count"] / len(df) * 100
    ).round(2)
    missing_report.to_csv(MISSING_VALUE_REPORT_PATH, index=False)
    return missing_report


def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically impossible PV and weather values."""
    clean_df = df.copy()

    # Drop rows where model input or target columns are missing.
    clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)

    # Keep values within realistic operating ranges for a small PV demo dataset.
    valid_ranges = (
        clean_df["irradiance"].between(0, 1200)
        & clean_df["ambient_temperature"].between(-30, 60)
        & clean_df["module_temperature"].between(-30, 90)
        & clean_df["humidity"].between(0, 100)
        & clean_df["wind_speed"].between(0, 40)
        & clean_df["hour"].between(0, 23)
        & clean_df["day_of_year"].between(1, 366)
        & clean_df["power_output"].between(0, 120)
    )
    clean_df = clean_df.loc[valid_ranges].copy()

    return clean_df


def remove_statistical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme statistical outliers using the IQR method."""
    clean_df = df.copy()
    columns_to_check = [
        "irradiance",
        "ambient_temperature",
        "module_temperature",
        "humidity",
        "wind_speed",
        "power_output",
    ]

    for column in columns_to_check:
        q1 = clean_df[column].quantile(0.25)
        q3 = clean_df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clean_df = clean_df[clean_df[column].between(lower_bound, upper_bound)]

    return clean_df.reset_index(drop=True)


def save_distribution_plot(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """Visualize raw and cleaned data distributions for key PV variables."""
    columns_to_plot = [
        "irradiance",
        "ambient_temperature",
        "module_temperature",
        "humidity",
        "wind_speed",
        "power_output",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, column in zip(axes, columns_to_plot):
        ax.hist(raw_df[column], bins=35, alpha=0.45, label="Raw")
        ax.hist(clean_df[column], bins=35, alpha=0.65, label="Clean")
        ax.set_title(column)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("PV Data Distribution Before and After Cleaning")
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(RAW_DATA_PATH)
    validate_columns(raw_df)

    missing_report = save_missing_value_report(raw_df)
    range_clean_df = remove_invalid_values(raw_df)
    clean_df = remove_statistical_outliers(range_clean_df)

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    save_distribution_plot(raw_df, clean_df)

    print("Missing value report:")
    print(missing_report.to_string(index=False))
    print(f"Raw rows: {len(raw_df):,}")
    print(f"Clean rows: {len(clean_df):,}")
    print(f"Removed rows: {len(raw_df) - len(clean_df):,}")
    print(f"Saved clean dataset: {CLEAN_DATA_PATH}")
    print(f"Saved distribution plot: {DISTRIBUTION_PLOT_PATH}")
    print(f"Saved missing value report: {MISSING_VALUE_REPORT_PATH}")


if __name__ == "__main__":
    main()
