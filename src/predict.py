from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "pv_power_model.joblib"


def predict_power(sample: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run `python src/train_model.py` first."
        )

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    X = pd.DataFrame([sample], columns=feature_columns)
    prediction = float(model.predict(X)[0])
    return max(prediction, 0.0)


def main() -> None:
    sample = {
        "irradiance_w_m2": 780,
        "ambient_temp_c": 28,
        "module_temp_c": 47,
        "wind_speed_m_s": 2.5,
        "humidity_pct": 55,
        "cloud_cover_pct": 20,
        "hour": 12,
        "day_of_year": 180,
    }
    predicted_power = predict_power(sample)
    print(f"Predicted normal PV power: {predicted_power:.2f} kW")


if __name__ == "__main__":
    main()
