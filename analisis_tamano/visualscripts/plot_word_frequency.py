"""
Histogram: Word Frequency Distribution
-------------------------------------
Generates histogram of word occurrence counts from all inputs.
Data source: output/word_counts.csv
"""

import polars as pl
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
HISTOGRAM_DIR = os.path.join(OUTPUT_DIR, "histograms")

def main():
    df = pl.read_csv(os.path.join(OUTPUT_DIR, "word_counts.csv"))
    counts = df["count"].to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Word Frequency Distribution")
    ax.set_xlabel("Word Occurrence Count")
    ax.set_ylabel("Number of Words")
    ax.axvline(sum(counts)/len(counts), color="red", linestyle="--",
               label=f"Mean: {sum(counts)/len(counts):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HISTOGRAM_DIR, "word_frequency_distribution.png"), dpi=150)
    print("Saved: word_frequency_distribution.png")

if __name__ == "__main__":
    main()