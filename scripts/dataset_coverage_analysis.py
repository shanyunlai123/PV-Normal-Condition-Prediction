"""Build multi-site and multi-year datasets from standardized PVDAQ samples."""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
INPUT_PATH = PROCESSED_DIR / "real_pvdaq_standardized.csv"
MANIFEST_PATH = RAW_DIR / "pvdaq_site_manifest.csv"
MULTI_SITE_PATH = PROCESSED_DIR / "multi_site_dataset.csv"
MULTI_YEAR_PATH = PROCESSED_DIR / "multi_year_dataset.csv"
AUDIT_PATH = ROOT_DIR / "results" / "pvdaq_coverage_audit.csv"
EXCLUSION_PATH = ROOT_DIR / "results" / "pvdaq_candidate_exclusions.csv"

# Sites reviewed but not included because the required common research fields
# cannot be formed from their published channels.
EXCLUDED_CANDIDATES = [
    {
        "system_id": 1199,
        "reason": "No irradiance or temperature channels; inverter power only.",
    },
    {
        "system_id": 1200,
        "reason": "Metered AC power is available, but irradiance, temperature, voltage, and current are absent.",
    },
    {
        "system_id": 1332,
        "reason": "Utility-scale power/current channels are available, but irradiance and temperature are absent.",
    },
    {
        "system_id": 1430,
        "reason": "Irradiance and temperature are available, but no power channel is published.",
    },
]


def season_from_month(month: pd.Series) -> pd.Series:
    """Map calendar months to meteorological seasons."""
    return month.map(
        {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
    )


def main() -> None:
    """Create research datasets and document excluded sites/fields."""
    df = pd.read_csv(INPUT_PATH, parse_dates=["time"])
    manifest = pd.read_csv(MANIFEST_PATH)
    df = df.merge(
        manifest[["system_id", "system_public_name", "dc_capacity_kw", "site_category"]],
        on="system_id",
        how="left",
    )
    df["year"] = df["time"].dt.year
    df["date"] = df["time"].dt.date.astype("string")
    df["season"] = season_from_month(df["time"].dt.month)
    df["irradiance"] = df["irradiance"].clip(lower=0)
    df["power"] = df["power"].clip(lower=0)

    # Irradiance, temperature, and power are essential. Voltage/current remain optional.
    usable = df.dropna(subset=["time", "irradiance", "temperature", "power", "system_id"]).copy()
    usable = (
        usable.sort_values(["system_id", "time"])
        .drop_duplicates(subset=["system_id", "time"], keep="first")
        .reset_index(drop=True)
    )
    usable.to_csv(MULTI_SITE_PATH, index=False)
    usable[usable["year"].between(2020, 2023)].to_csv(MULTI_YEAR_PATH, index=False)

    audit_rows = []
    for _, site in manifest.iterrows():
        site_df = df[df["system_id"] == site["system_id"]]
        audit_rows.append(
            {
                "system_id": site["system_id"],
                "source_site": site["source_site"],
                "site_category": site["site_category"],
                "rows_raw_standardized": len(site_df),
                "rows_usable": site_df[
                    ["time", "irradiance", "temperature", "power"]
                ].notna().all(axis=1).sum(),
                "years": ",".join(map(str, sorted(site_df["year"].dropna().astype(int).unique()))),
                "missing_irradiance_percent": site_df["irradiance"].isna().mean() * 100 if len(site_df) else 100,
                "missing_temperature_percent": site_df["temperature"].isna().mean() * 100 if len(site_df) else 100,
                "missing_voltage_percent": site_df["voltage"].isna().mean() * 100 if len(site_df) else 100,
                "missing_current_percent": site_df["current"].isna().mean() * 100 if len(site_df) else 100,
                "missing_power_percent": site_df["power"].isna().mean() * 100 if len(site_df) else 100,
            }
        )
    pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
    pd.DataFrame(EXCLUDED_CANDIDATES).to_csv(EXCLUSION_PATH, index=False)

    print(f"Multi-site rows: {len(usable):,}; sites: {usable['system_id'].nunique()}")
    print(f"Years: {sorted(usable['year'].unique())}")
    print(f"Seasons: {sorted(usable['season'].dropna().unique())}")
    print(f"Saved: {MULTI_SITE_PATH}")
    print(f"Saved: {MULTI_YEAR_PATH}")
    print(f"Saved: {AUDIT_PATH}")
    print(f"Saved: {EXCLUSION_PATH}")


if __name__ == "__main__":
    main()
