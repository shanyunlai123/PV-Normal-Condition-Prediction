from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_pv_power_model.joblib"


def predict_power(sample: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run `python src/train_model.py` first."
        )

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    X = pd.DataFrame([sample], columns=feature_columns)
    return max(float(model.predict(X)[0]), 0.0)


def main() -> None:
    sample = {
        "irradiance": 780,
        "ambient_temperature": 28,
        "module_temperature": 47,
        "humidity": 55,
        "wind_speed": 2.5,
        "hour": 12,
        "day_of_year": 180,
    }

    predicted_power = predict_power(sample)
    print(f"Predicted normal PV power output: {predicted_power:.2f} kW")


if __name__ == "__main__":
    main()
