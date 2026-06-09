"""Create extended PV operating scenario datasets from combined real PVDAQ data."""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "real_pvdaq_combined.csv"

OUTPUT_PATHS = {
    "sunny": PROCESSED_DIR / "sunny_dataset.csv",
    "cloudy": PROCESSED_DIR / "cloudy_dataset.csv",
    "moderate": PROCESSED_DIR / "moderate_dataset.csv",
    "rainy": PROCESSED_DIR / "rainy_dataset.csv",
    "high_temperature": PROCESSED_DIR / "high_temperature_dataset.csv",
    "low_temperature": PROCESSED_DIR / "low_temperature_dataset.csv",
    "all_weather": PROCESSED_DIR / "all_weather_dataset.csv",
}


def main() -> None:
    """Classify and save weather and temperature operating scenarios."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing combined data: {INPUT_PATH}. Run `python scripts/merge_real_pvdaq_data.py` first."
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["time"])
    df = df.sort_values(["source_site", "time"]).reset_index(drop=True)

    # A rolling standard deviation provides a simple irradiance fluctuation measure.
    df["irradiance_rolling_std"] = (
        df.groupby("source_site")["irradiance"]
        .transform(lambda series: series.rolling(window=15, min_periods=5).std())
        .fillna(0)
    )

    daytime = df[df["irradiance"] >= 20].copy()
    fluctuation_threshold = float(daytime["irradiance_rolling_std"].quantile(0.75))
    high_temperature_threshold = float(df["temperature"].quantile(0.75))
    low_temperature_threshold = float(df["temperature"].quantile(0.25))

    scenarios = {
        "sunny": df[df["irradiance"] >= 600].copy(),
        "cloudy": df[df["irradiance"].between(20, 250, inclusive="left")].copy(),
        "moderate": df[df["irradiance"].between(250, 600, inclusive="left")].copy(),
        # Rainy is a proxy because PVDAQ samples do not contain a rain sensor.
        "rainy": df[
            (df["irradiance"].between(20, 300, inclusive="left"))
            & (df["irradiance_rolling_std"] >= fluctuation_threshold)
        ].copy(),
        "high_temperature": df[df["temperature"] >= high_temperature_threshold].copy(),
        "low_temperature": df[df["temperature"] <= low_temperature_threshold].copy(),
        "all_weather": df.copy(),
    }

    summary_rows = []
    for scenario_name, scenario_df in scenarios.items():
        scenario_df["operating_scenario"] = scenario_name
        scenario_df.to_csv(OUTPUT_PATHS[scenario_name], index=False)
        summary_rows.append(
            {
                "scenario": scenario_name,
                "rows": len(scenario_df),
                "temperature_threshold_low": low_temperature_threshold,
                "temperature_threshold_high": high_temperature_threshold,
                "rainy_fluctuation_threshold": fluctuation_threshold,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ROOT_DIR / "results" / "operating_scenario_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Operating scenario summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved scenario summary: {summary_path}")


if __name__ == "__main__":
    main()
