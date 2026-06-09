"""Create a correlation matrix and heatmap for real PVDAQ model variables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "all_weather_dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
MATRIX_PATH = RESULTS_DIR / "correlation_matrix.csv"
HEATMAP_PATH = RESULTS_DIR / "correlation_heatmap.png"

FEATURES = ["irradiance", "temperature", "voltage", "current", "power"]


def main() -> None:
    """Calculate Pearson correlations and create a matplotlib heatmap."""
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES)
    correlation = df[FEATURES].corr(method="pearson")
    correlation.to_csv(MATRIX_PATH)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)

    for row in range(len(FEATURES)):
        for column in range(len(FEATURES)):
            value = correlation.iloc[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center")

    ax.set_title("Correlation Matrix for Real PVDAQ Variables")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=160)
    plt.close()

    print("Correlation with power:")
    print(correlation["power"].sort_values(ascending=False).to_string())
    print(f"Saved correlation matrix: {MATRIX_PATH}")
    print(f"Saved correlation heatmap: {HEATMAP_PATH}")


if __name__ == "__main__":
    main()
