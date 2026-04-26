"""
Chatbot Interaction Analysis
----------------------------
Analyzes customer support conversations from dataset_50k_anonymized.
Generates per-conversation statistics and global histograms.
"""

import polars as pl
import matplotlib.pyplot as plt
import os
import sys

try:
    import importlib
    assert importlib.util.find_spec("polars") is not None
except AssertionError:
    print("Error: 'polars' package not found. Install with: pip install polars")
    sys.exit(1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_50k_anonymized.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
HISTOGRAM_DIR = os.path.join(OUTPUT_DIR, "histograms")


def load_data(path: str) -> pl.DataFrame:
    """Load parquet dataset with error handling."""
    try:
        df = pl.read_parquet(path)
        print(f"Loaded {len(df):,} rows from {path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading parquet: {e}")
        sys.exit(1)


def compute_conversation_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate per-conversation statistics.

    For each conv_id, compute:
    - n_turns: number of interactions
    - input_length_mean: average character length of user messages
    - output_length_mean: average character length of assistant responses
    - input_length_total: total input characters across all turns
    - output_length_total: total output characters across all turns
    - total_length: combined input + output
    - channel_mode: most frequent channel_source value
    """
    try:
        convo_stats = (
            df.group_by("conv_id")
            .agg(
                [
                    pl.len().alias("n_turns"),
                    pl.col("input").str.len_bytes().mean().alias("input_length_mean"),
                    pl.col("output").str.len_bytes().mean().alias("output_length_mean"),
                    pl.col("input").str.len_bytes().sum().alias("input_length_total"),
                    pl.col("output").str.len_bytes().sum().alias("output_length_total"),
                    (
                        pl.col("input").str.len_bytes().sum()
                        + pl.col("output").str.len_bytes().sum()
                    ).alias("total_length"),
                    pl.col("channel_source").mode().first().alias("channel_mode"),
                ]
            )
        )
        print(f"Computed stats for {len(convo_stats):,} conversations")
        return convo_stats
    except Exception as e:
        print(f"Error computing stats: {e}")
        sys.exit(1)


def plot_histograms(df: pl.DataFrame, output_dir: str) -> None:
    """
    Generate three PNG histograms showing distribution of per-conversation stats:
    1. Mean input length
    2. Mean output length
    3. Total conversation length
    """
    metrics = [
        ("input_length_mean", "Input Length Mean (bytes)", "input_length_distribution.png"),
        ("output_length_mean", "Output Length Mean (bytes)", "output_length_distribution.png"),
        ("total_length", "Total Conversation Length (bytes)", "total_length_distribution.png"),
    ]

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (col, title, filename) in zip(axes, metrics):
            data = df[col].drop_nulls().to_list()
            ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Bytes")
            ax.set_ylabel("Conversation Count")
            ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.0f}")
            ax.axvline(df[col].median(), color="green", linestyle="--", label=f"Median: {df[col].median():.0f}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename), dpi=150)
            print(f"Saved histogram: {filename}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating histograms: {e}")
        sys.exit(1)


def export_stats(df: pl.DataFrame, output_path: str) -> None:
    """Export per-conversation stats to CSV."""
    try:
        df.write_csv(output_path)
        print(f"Exported stats to {output_path}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        sys.exit(1)


def print_summary(df: pl.DataFrame) -> None:
    """Print aggregate summary statistics."""
    print("\n=== Aggregate Statistics ===")
    print(f"Total conversations: {len(df):,}")

    cols = ["n_turns", "input_length_mean", "output_length_mean", "total_length"]
    for col in cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean:   {df[col].mean():.2f}")
            print(f"  Median: {df[col].median():.2f}")
            print(f"  Std:    {df[col].std():.2f}")
            print(f"  Min:    {df[col].min():.2f}")
            print(f"  Max:    {df[col].max():.2f}")


def main():
    df = load_data(DATA_PATH)
    convo_stats = compute_conversation_stats(df)
    plot_histograms(convo_stats, HISTOGRAM_DIR)
    export_stats(convo_stats, os.path.join(OUTPUT_DIR, "stats.csv"))
    print_summary(convo_stats)


if __name__ == "__main__":
    main()