"""Download a reproducible multi-site, multi-year PVDAQ research sample."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw" / "real_pvdaq"
MANIFEST_PATH = ROOT_DIR / "data" / "raw" / "pvdaq_site_manifest.csv"
BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/"
BUCKET_LIST_URL = f"{BASE_URL}?"
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Capacity-based research categories. These are not official PVDAQ labels.
SELECTED_SITES = {
    2: ("Residential 1a", 2.912, "residential_proxy"),
    4: ("NREL x-Si -1", 1.000, "small_experimental"),
    10: ("NREL CIS -1", 1.120, "small_experimental"),
    33: ("Silicor Materials", 2.400, "small_experimental"),
    50: ("NREL x-Si 6", 6.000, "small_experimental"),
    34: ("Andre Agassi Preparatory Academy - Building A", 146.640, "commercial"),
    14200: ("SAS_SF1", 1000.000, "utility_scale"),
    14201: ("SAS_SF2", 1340.000, "utility_scale"),
}
TARGET_YEARS = [2020, 2021, 2022, 2023]
TARGET_DATES = ["01-15", "04-15", "07-15", "10-15"]


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """List all public S3 objects below a prefix."""
    query = urllib.parse.urlencode({"prefix": prefix, "max-keys": 1000})
    root = ET.fromstring(urllib.request.urlopen(BUCKET_LIST_URL + query, timeout=60).read())
    return [
        (
            item.find("s3:Key", S3_NAMESPACE).text,
            int(item.find("s3:Size", S3_NAMESPACE).text),
        )
        for item in root.findall("s3:Contents", S3_NAMESPACE)
    ]


def download_csv(key: str) -> pd.DataFrame:
    """Download one public PVDAQ CSV object."""
    payload = urllib.request.urlopen(BASE_URL + key, timeout=180).read()
    return pd.read_csv(BytesIO(payload))


def select_daily_keys(system_id: int, year: int) -> list[str]:
    """Select four seasonal dates, falling back to the first available day in each month."""
    selected = []
    for month_day in TARGET_DATES:
        month, day = month_day.split("-")
        prefix = (
            f"pvdaq/csv/pvdata/system_id={system_id}/year={year}/"
            f"month={int(month)}/day={int(day)}/"
        )
        date_keys = [key for key, size in list_keys(prefix) if size > 0 and key.endswith(".csv")]
        if not date_keys:
            month_prefix = (
                f"pvdaq/csv/pvdata/system_id={system_id}/year={year}/month={int(month)}/"
            )
            date_keys = [
                key for key, size in list_keys(month_prefix) if size > 0 and key.endswith(".csv")
            ][:1]
        selected.extend(date_keys)
    return selected


def download_site_year(system_id: int, year: int) -> pd.DataFrame | None:
    """Download seasonal daily files or sample an annual-format file."""
    daily_keys = select_daily_keys(system_id, year)
    if daily_keys:
        frames = [frame for frame in (download_csv(key) for key in daily_keys) if not frame.empty]
        if frames:
            return pd.concat(frames, ignore_index=True)

    annual_prefix = f"pvdaq/csv/pvdata/system_id={system_id}/year={year}/"
    annual_keys = [
        key for key, size in list_keys(annual_prefix) if size > 0 and key.endswith(".csv")
    ]
    if not annual_keys:
        return None

    # Daily-format sites may have empty files near their final timestamp. Keep the
    # first non-empty day as a documented fallback when seasonal dates are empty.
    if "__date_" in annual_keys[0]:
        for key in annual_keys:
            fallback = download_csv(key)
            if not fallback.empty:
                return fallback
        return None

    annual = pd.concat([download_csv(key) for key in annual_keys], ignore_index=True)
    time_column = "utc_measured_on" if "utc_measured_on" in annual.columns else "measured_on"
    timestamps = pd.to_datetime(annual[time_column], errors="coerce")
    selected_dates = {f"{year}-{month_day}" for month_day in TARGET_DATES}
    return annual[timestamps.dt.strftime("%Y-%m-%d").isin(selected_dates)].copy()


def main() -> None:
    """Download selected public PVDAQ samples and write a site manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for system_id, (name, capacity_kw, category) in SELECTED_SITES.items():
        downloaded_years = []
        for year in TARGET_YEARS:
            output_path = RAW_DIR / f"system_{system_id}_{year}_seasonal_sample.csv"
            if output_path.exists():
                sample = pd.read_csv(output_path)
            else:
                print(f"Downloading PVDAQ system {system_id}, year {year}...")
                sample = download_site_year(system_id, year)
                if sample is None or sample.empty:
                    print(f"  No selected data available for {year}")
                    continue
                sample["system_id"] = system_id
                sample.to_csv(output_path, index=False)
            downloaded_years.append(year)
            print(f"  Saved {len(sample):,} rows: {output_path.name}")

        manifest_rows.append(
            {
                "system_id": system_id,
                "source_site": f"PVDAQ_{system_id}",
                "system_public_name": name,
                "dc_capacity_kw": capacity_kw,
                "site_category": category,
                "downloaded_years": ",".join(map(str, downloaded_years)),
                "source": "NREL PVDAQ OEDI public data lake",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
    print(f"Saved site manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
