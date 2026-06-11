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
    "time": ["measured_on", "utc_measured_on", "timestamp", "time"],
    "irradiance": ["poa_irradiance", "irradiance_poa", "ghi", "irradiance_ghi", "irradiance"],
    "temperature": [
        "ambient_temp",
        "temperature_ambient",
        "module_temp_1",
        "module_temp",
        "module_temp_f",
        "temperature_module",
        "temperature",
    ],
    "voltage": ["ac_voltage", "dc_pos_voltage", "dc_voltage", "voltage"],
    "current": ["ac_current", "dc_pos_current", "dc_current_meter", "dc_current", "current"],
    "power": ["ac_power_meter", "ac_power_hw", "ac_power", "dc_power", "power"],
}


def find_columns(columns: list[str], prefixes: list[str]) -> list[str]:
    """Find columns matching the first available sensor prefix."""
    for prefix in prefixes:
        matches = [
            column
            for column in columns
            if column == prefix
            or column.startswith(f"{prefix}__")
            or column.startswith(f"{prefix}_inv_")
            or column.startswith(f"{prefix}_")
        ]
        if matches:
            return matches
    return []


def combine_sensor_columns(raw_df: pd.DataFrame, field: str, columns: list[str]) -> pd.Series:
    """Combine multi-inverter sensors using physically appropriate aggregation."""
    if not columns:
        return pd.Series(np.nan, index=raw_df.index)
    numeric = raw_df[columns].apply(pd.to_numeric, errors="coerce")
    if field in {"power", "current"} and len(columns) > 1:
        return numeric.sum(axis=1, min_count=1)
    if field in {"voltage", "temperature", "irradiance"} and len(columns) > 1:
        return numeric.mean(axis=1)
    return numeric.iloc[:, 0]


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
        field: find_columns(columns, prefixes)
        for field, prefixes in FIELD_PREFIXES.items()
    }

    standardized = pd.DataFrame(index=raw_df.index)
    time_columns = mapped_columns["time"]
    standardized["time"] = raw_df[time_columns[0]] if time_columns else np.nan
    for field in ["irradiance", "temperature", "voltage", "current", "power"]:
        standardized[field] = combine_sensor_columns(raw_df, field, mapped_columns[field])

    standardized["time"] = pd.to_datetime(standardized["time"], errors="coerce")
    for field in ["irradiance", "temperature", "voltage", "current", "power"]:
        standardized[field] = pd.to_numeric(standardized[field], errors="coerce")

    if "system_id" in raw_df.columns:
        standardized["system_id"] = pd.to_numeric(raw_df["system_id"], errors="coerce")
        standardized["source_site"] = "PVDAQ_" + standardized["system_id"].astype("Int64").astype(str)
    else:
        standardized["system_id"] = pd.NA
        standardized["source_site"] = path.stem.split("_202")[0]

    standardized["weather_condition"] = basic_weather_label(standardized["irradiance"])
    standardized["source_file"] = path.name

    missing = [field for field, source in mapped_columns.items() if not source]
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
