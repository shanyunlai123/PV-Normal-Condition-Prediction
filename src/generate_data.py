from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "simulated_pv_data.csv"


def generate_simulated_pv_data(
    start: str = "2024-01-01",
    periods: int = 365 * 24,
    freq: str = "h",
    capacity_kw: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hourly normal-condition PV power data for one year."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)

    hour = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    daylight_shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal_shape = 0.72 + 0.28 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    random_cloud_effect = 1 - rng.beta(2.0, 5.0, periods) * 0.55

    irradiance = 1000 * daylight_shape * seasonal_shape * random_cloud_effect
    irradiance = np.clip(irradiance + rng.normal(0, 25, periods), 0, 1100)

    ambient_temperature = (
        16
        + 10 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
        + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 2, periods)
    )
    wind_speed = np.clip(rng.gamma(shape=2.0, scale=1.2, size=periods), 0, 12)
    humidity = np.clip(
        70 - 0.45 * ambient_temperature + rng.normal(0, 8, periods),
        15,
        100,
    )
    module_temperature = ambient_temperature + irradiance * 0.03 - wind_speed * 0.8

    temperature_loss = 1 - 0.004 * (module_temperature - 25)
    expected_power = capacity_kw * (irradiance / 1000) * temperature_loss * 0.965
    power_output = np.clip(expected_power + rng.normal(0, capacity_kw * 0.015, periods), 0, capacity_kw)
    power_output[irradiance < 20] = 0

    return pd.DataFrame(
        {
            "irradiance": np.round(irradiance, 2),
            "ambient_temperature": np.round(ambient_temperature, 2),
            "module_temperature": np.round(module_temperature, 2),
            "humidity": np.round(humidity, 2),
            "wind_speed": np.round(wind_speed, 2),
            "hour": hour,
            "day_of_year": day_of_year,
            "power_output": np.round(power_output, 3),
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_simulated_pv_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated simulated PV dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
