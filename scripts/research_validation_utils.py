"""Shared helpers for cross-site and temporal PVDAQ validation."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "multi_year_dataset.csv"
FEATURES = ["irradiance", "temperature"]
TARGET = "normalized_power"


def build_model() -> RandomForestRegressor:
    """Use the existing Random Forest family with fixed, reproducible settings."""
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )


def load_validation_data() -> pd.DataFrame:
    """Load data and normalize mixed W/kW power values by site capacity."""
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    site_max = df.groupby("system_id")["power"].transform("max")
    # PVDAQ public systems use both W and kW. Values over 10x rated kW are treated as W.
    df["power_unit_divisor"] = (site_max > df["dc_capacity_kw"] * 10).map(
        {True: 1000.0, False: 1.0}
    )
    df["power_kw"] = df["power"] / df["power_unit_divisor"]
    df[TARGET] = df["power_kw"] / df["dc_capacity_kw"]

    # Keep physically plausible daytime normal-operation rows.
    clean = df.dropna(subset=FEATURES + [TARGET, "system_id", "year", "season"]).copy()
    clean = clean[
        (clean["irradiance"] >= 20)
        & (clean[TARGET] >= 0)
        & (clean[TARGET] <= 1.5)
    ]
    return clean.reset_index(drop=True)
