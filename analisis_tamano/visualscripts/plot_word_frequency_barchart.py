"""
Bar Chart: Word Frequency (Top N)
---------------------------------
Displays top N most frequent words as a horizontal bar chart.
Excludes words listed in filtered_words.txt.
"""

import polars as pl
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
HISTOGRAM_DIR = os.path.join(OUTPUT_DIR, "histograms")
FILTER_FILE = os.path.join(OUTPUT_DIR, "filtered_words.txt")

TOP_N = 20  # Number of top words to display

def load_filtered_words(path: str) -> set:
    """Load words to exclude from file, one per line."""
    if not os.path.exists(path):
        print(f"Filter file not found: {path} (no exclusions)")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        words = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded {len(words)} words to exclude from {path}")
    return words

def main():
    excluded = load_filtered_words(FILTER_FILE)

    df = pl.read_csv(os.path.join(OUTPUT_DIR, "word_counts.csv"))

    if excluded:
        df = df.filter(~pl.col("word").str.to_lowercase().is_in(excluded))
        print(f"After filtering: {len(df):,} words")

    top_words = df.top_k(TOP_N, by="count")

    fig, ax = plt.subplots(figsize=(10, max(6, TOP_N * 0.4)))
    ax.barh(range(len(top_words)), top_words["count"].to_list(), color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words["word"].to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {TOP_N} Most Frequent Words")
    for i, count in enumerate(top_words["count"]):
        ax.text(count + 50, i, f"{count}", va="center", fontsize=9)
    ax.set_xlim(0, top_words["count"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(HISTOGRAM_DIR, "word_frequency_barchart.png"), dpi=150)
    print(f"Saved: word_frequency_barchart.png ({TOP_N} words)")

if __name__ == "__main__":
    main()