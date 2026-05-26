import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "simulated_pv_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "pv_power_model.joblib"
METRICS_PATH = ROOT_DIR / "results" / "metrics.json"
PREDICTION_PLOT_PATH = ROOT_DIR / "results" / "prediction_vs_actual.png"
IMPORTANCE_PLOT_PATH = ROOT_DIR / "results" / "feature_importance.png"

FEATURE_COLUMNS = [
    "irradiance_w_m2",
    "ambient_temp_c",
    "module_temp_c",
    "wind_speed_m_s",
    "humidity_pct",
    "cloud_cover_pct",
    "hour",
    "day_of_year",
]
TARGET_COLUMN = "power_kw"


def build_model():
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        return model, "xgboost"
    except ImportError:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=14,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return model, "random_forest"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file: {DATA_PATH}. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()


def save_prediction_plot(y_test: pd.Series, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=14)
    limit = max(float(y_test.max()), float(y_pred.max()))
    plt.plot([0, limit], [0, limit], color="red", linewidth=2)
    plt.xlabel("Actual power (kW)")
    plt.ylabel("Predicted power (kW)")
    plt.title("PV Power: Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=160)
    plt.close()


def save_feature_importance_plot(model, model_type: str) -> None:
    if model_type == "xgboost":
        importances = model.feature_importances_
    else:
        importances = model.named_steps["regressor"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(len(FEATURE_COLUMNS))
    )

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "results").mkdir(exist_ok=True)

    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, model_type = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "model_type": model_type,
        "rows": int(len(df)),
        "mae_kw": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse_kw": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }

    joblib.dump(
        {
            "model": model,
            "model_type": model_type,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(y_test, y_pred)
    save_feature_importance_plot(model, model_type)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
