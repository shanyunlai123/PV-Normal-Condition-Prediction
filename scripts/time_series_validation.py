"""Compare random K-Fold validation with future-year temporal validation."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from research_validation_utils import FEATURES, TARGET, build_model, load_validation_data


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "results" / "time_series_validation.csv"


def metric_row(method: str, train_years: str, test_year: str, actual, prediction) -> dict:
    """Build one comparable validation result row."""
    return {
        "validation_method": method,
        "train_years": train_years,
        "test_year": test_year,
        "test_rows": len(actual),
        "mae": mean_absolute_error(actual, prediction),
        "rmse": mean_squared_error(actual, prediction) ** 0.5,
        "r2": r2_score(actual, prediction),
    }


def main() -> None:
    """Run random and expanding-window temporal validation."""
    df = load_validation_data().sort_values("time").reset_index(drop=True)
    random_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    random_prediction = cross_val_predict(
        build_model(), df[FEATURES], df[TARGET], cv=random_cv, n_jobs=-1
    )
    rows = [
        metric_row(
            "Random KFold",
            "mixed 2020-2023",
            "mixed 2020-2023",
            df[TARGET],
            random_prediction,
        )
    ]

    for test_year in sorted(df["year"].unique()):
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        if train.empty or len(test) < 25:
            continue
        model = build_model()
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(test[FEATURES])
        rows.append(
            metric_row(
                "Time-Series Expanding Window",
                f"{int(train['year'].min())}-{int(train['year'].max())}",
                str(int(test_year)),
                test[TARGET],
                prediction,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
