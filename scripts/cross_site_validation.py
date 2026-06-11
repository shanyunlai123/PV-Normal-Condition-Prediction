"""Evaluate PV prediction transfer to completely unseen PVDAQ sites."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from research_validation_utils import FEATURES, TARGET, build_model, load_validation_data


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "results" / "cross_site_validation.csv"


def main() -> None:
    """Run leave-one-site-out validation using normalized power."""
    df = load_validation_data()
    rows = []

    for system_id in sorted(df["system_id"].unique()):
        train = df[df["system_id"] != system_id]
        test = df[df["system_id"] == system_id]
        if len(test) < 25:
            continue

        model = build_model()
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(test[FEATURES])
        rows.append(
            {
                "test_system_id": int(system_id),
                "source_site": test["source_site"].iloc[0],
                "site_category": test["site_category"].iloc[0],
                "train_sites": train["system_id"].nunique(),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_years": ",".join(map(str, sorted(test["year"].unique()))),
                "mae": mean_absolute_error(test[TARGET], prediction),
                "rmse": mean_squared_error(test[TARGET], prediction) ** 0.5,
                "r2": r2_score(test[TARGET], prediction),
            }
        )

    results = pd.DataFrame(rows).sort_values("rmse")
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
