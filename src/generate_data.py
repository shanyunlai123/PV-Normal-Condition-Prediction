from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "simulated_pv_data.csv"


def generate_simulated_pv_data(
    start: str = "2024-01-01",
    periods: int = 365 * 24,
    freq: str = "h",
    capacity_kw: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate normal-condition PV power data with realistic weather patterns."""
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range(start=start, periods=periods, freq=freq)

    hour = timestamp.hour.to_numpy()
    day_of_year = timestamp.dayofyear.to_numpy()

    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal = 0.72 + 0.28 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    cloud_cover_pct = np.clip(rng.beta(2.0, 5.0, periods) * 100, 0, 100)
    cloud_loss = 1 - 0.75 * (cloud_cover_pct / 100) ** 1.4

    irradiance_w_m2 = 1000 * daylight * seasonal * cloud_loss
    irradiance_w_m2 += rng.normal(0, 25, periods)
    irradiance_w_m2 = np.clip(irradiance_w_m2, 0, 1100)

    ambient_temp_c = (
        15
        + 10 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
        + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 2.0, periods)
    )
    wind_speed_m_s = np.clip(rng.gamma(shape=2.0, scale=1.2, size=periods), 0, 12)
    humidity_pct = np.clip(
        65 - 0.25 * ambient_temp_c + 0.25 * cloud_cover_pct + rng.normal(0, 8, periods),
        15,
        100,
    )
    module_temp_c = ambient_temp_c + irradiance_w_m2 * 0.028 - wind_speed_m_s * 0.7

    temp_coefficient = -0.004
    temp_factor = 1 + temp_coefficient * (module_temp_c - 25)
    inverter_efficiency = 0.965
    expected_power_kw = capacity_kw * (irradiance_w_m2 / 1000) * temp_factor * inverter_efficiency

    noise = rng.normal(0, capacity_kw * 0.015, periods)
    power_kw = np.clip(expected_power_kw + noise, 0, capacity_kw)
    power_kw[irradiance_w_m2 < 20] = 0

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "irradiance_w_m2": np.round(irradiance_w_m2, 2),
            "ambient_temp_c": np.round(ambient_temp_c, 2),
            "module_temp_c": np.round(module_temp_c, 2),
            "wind_speed_m_s": np.round(wind_speed_m_s, 2),
            "humidity_pct": np.round(humidity_pct, 2),
            "cloud_cover_pct": np.round(cloud_cover_pct, 2),
            "hour": hour,
            "day_of_year": day_of_year,
            "power_kw": np.round(power_kw, 3),
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_simulated_pv_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
