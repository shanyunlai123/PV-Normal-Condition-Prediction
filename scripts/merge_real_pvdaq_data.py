"""Merge standardized multi-site, multi-date PVDAQ data."""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
STANDARDIZED_PATH = PROCESSED_DIR / "real_pvdaq_standardized.csv"
OUTPUT_PATH = PROCESSED_DIR / "real_pvdaq_combined.csv"


def main() -> None:
    """Add date metadata, remove unusable rows, and save the combined real dataset."""
    if not STANDARDIZED_PATH.exists():
        raise FileNotFoundError(
            f"Missing standardized data: {STANDARDIZED_PATH}. "
            "Run `python scripts/standardize_pvdaq_data.py` first."
        )

    df = pd.read_csv(STANDARDIZED_PATH, parse_dates=["time"])
    df["date"] = df["time"].dt.date.astype("string")

    # Keep rows that have the three essential prediction variables.
    combined = df.dropna(subset=["time", "irradiance", "temperature", "power"]).copy()

    # Negative nighttime sensor noise is not useful for normal power prediction.
    combined["irradiance"] = combined["irradiance"].clip(lower=0)
    combined["power"] = combined["power"].clip(lower=0)
    combined = combined.sort_values(["source_site", "time"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    summary = combined.groupby(["source_site", "date"]).size().reset_index(name="rows")
    print("Combined real PVDAQ coverage:")
    print(summary.to_string(index=False))
    print(f"Saved combined PVDAQ data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
