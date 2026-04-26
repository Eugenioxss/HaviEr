"""
Histogram: Input Length Distribution
-------------------------------------
Generates histogram of per-conversation mean input length (in bytes).
Data source: output/stats.csv
"""

import polars as pl
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
HISTOGRAM_DIR = os.path.join(OUTPUT_DIR, "histograms")

def main():
    df = pl.read_csv(os.path.join(OUTPUT_DIR, "stats.csv"))
    data = df["input_length_mean"].drop_nulls().to_list()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Input Length Mean Distribution")
    ax.set_xlabel("Bytes")
    ax.set_ylabel("Conversation Count")
    ax.axvline(df["input_length_mean"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['input_length_mean'].mean():.0f}")
    ax.axvline(df["input_length_mean"].median(), color="green", linestyle="--",
               label=f"Median: {df['input_length_mean'].median():.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HISTOGRAM_DIR, "input_length_distribution.png"), dpi=150)
    print("Saved: input_length_distribution.png")

if __name__ == "__main__":
    main()