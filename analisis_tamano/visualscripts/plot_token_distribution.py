"""
Histogram: Token Distribution
-----------------------------
Generates histogram of per-conversation token counts (tiktoken cl100k_base).
Data source: output/token_stats.csv
"""

import polars as pl
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
HISTOGRAM_DIR = os.path.join(OUTPUT_DIR, "histograms")

def main():
    df = pl.read_csv(os.path.join(OUTPUT_DIR, "token_stats.csv"))
    data = df["total_tokens"].to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Total Tokens per Conversation")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.axvline(df["total_tokens"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['total_tokens'].mean():.0f}")
    ax.axvline(df["total_tokens"].median(), color="green", linestyle="--",
               label=f"Median: {df['total_tokens'].median():.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HISTOGRAM_DIR, "token_distribution.png"), dpi=150)
    print("Saved: token_distribution.png")

if __name__ == "__main__":
    main()