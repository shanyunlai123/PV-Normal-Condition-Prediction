"""Standardize raw PVDAQ CSV files into one beginner-friendly schema."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw" / "real_pvdaq"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "real_pvdaq_standardized.csv"

STANDARD_COLUMNS = [
    "time",
    "irradiance",
    "temperature",
    "voltage",
    "current",
    "power",
    "source_site",
    "weather_condition",
]

# PVDAQ sensor suffixes differ between sites, so prefixes are used for mapping.
FIELD_PREFIXES = {
    "time": ["measured_on", "timestamp", "time"],
    "irradiance": ["poa_irradiance", "ghi", "irradiance"],
    "temperature": ["ambient_temp", "module_temp_1", "temperature"],
    "voltage": ["ac_voltage", "dc_pos_voltage", "dc_voltage", "voltage"],
    "current": ["ac_current", "dc_pos_current", "dc_current", "current"],
    "power": ["ac_power", "dc_power", "power"],
}


def find_column(columns: list[str], prefixes: list[str]) -> str | None:
    """Find the first column whose name equals or starts with a known prefix."""
    for prefix in prefixes:
        for column in columns:
            if column == prefix or column.startswith(f"{prefix}__"):
                return column
    return None


def basic_weather_label(irradiance: pd.Series) -> pd.Series:
    """Create an initial weather label from irradiance; detailed scenarios are created later."""
    conditions = [
        irradiance < 20,
        irradiance.between(20, 250, inclusive="left"),
        irradiance.between(250, 600, inclusive="left"),
        irradiance >= 600,
    ]
    labels = ["night_or_low_power", "cloudy", "moderate", "sunny"]
    return pd.Series(np.select(conditions, labels, default="unknown"), index=irradiance.index)


def standardize_file(path: Path) -> pd.DataFrame:
    """Map one raw PVDAQ file to the standard schema without failing on missing sensors."""
    raw_df = pd.read_csv(path)
    columns = raw_df.columns.tolist()
    mapped_columns = {
        field: find_column(columns, prefixes)
        for field, prefixes in FIELD_PREFIXES.items()
    }

    standardized = pd.DataFrame(index=raw_df.index)
    for field in ["time", "irradiance", "temperature", "voltage", "current", "power"]:
        source_column = mapped_columns[field]
        standardized[field] = raw_df[source_column] if source_column else np.nan

    standardized["time"] = pd.to_datetime(standardized["time"], errors="coerce")
    for field in ["irradiance", "temperature", "voltage", "current", "power"]:
        standardized[field] = pd.to_numeric(standardized[field], errors="coerce")

    if "system_id" in raw_df.columns:
        standardized["source_site"] = "PVDAQ_" + raw_df["system_id"].astype(str)
    else:
        standardized["source_site"] = path.stem.split("_202")[0]

    standardized["weather_condition"] = basic_weather_label(standardized["irradiance"])
    standardized["source_file"] = path.name

    missing = [field for field, source in mapped_columns.items() if source is None]
    print(f"{path.name}: mapped={mapped_columns}, missing={missing or 'none'}")
    return standardized


def main() -> None:
    """Standardize every raw PVDAQ CSV and save one combined standardized file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw PVDAQ CSV files found in {RAW_DIR}")

    standardized_df = pd.concat(
        [standardize_file(path) for path in csv_files],
        ignore_index=True,
    )
    standardized_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Standardized files: {len(csv_files)}")
    print(f"Standardized rows: {len(standardized_df):,}")
    print(f"Saved standardized PVDAQ data: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
