"""
Token Context Size Estimation
----------------------------
Analyzes conversation token counts using tiktoken (cl100k_base).
Helps determine appropriate context window size for LLM training.
"""

import polars as pl
import tiktoken
import os
import sys
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_50k_anonymized.parquet")
STATS_PATH = os.path.join(BASE_DIR, "..", "output", "stats.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

SAMPLE_FRACTION = 0.10
WINDOWS = [4096, 8192, 16384, 32768]


def load_tokenizer():
    """Initialize tiktoken encoder."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        print("Loaded tiktoken encoder (cl100k_base)")
        return enc
    except Exception as e:
        print(f"Error loading tiktoken: {e}")
        sys.exit(1)


def sample_conversation_ids(stats_path: str, fraction: float) -> list:
    """Randomly sample conversation IDs from stats.csv."""
    try:
        df = pl.read_csv(stats_path)
        total = len(df)
        sample_size = int(total * fraction)
        sampled = df.sample(n=sample_size, seed=42)
        conv_ids = sampled["conv_id"].to_list()
        print(f"Sampled {len(conv_ids):,} conversations ({sample_size}/{total})")
        return conv_ids, df
    except Exception as e:
        print(f"Error loading stats: {e}")
        sys.exit(1)


def load_and_filter_parquet(data_path: str, conv_ids: list) -> pl.DataFrame:
    """Load parquet and filter to sampled conversations."""
    try:
        df = pl.read_parquet(data_path)
        filtered = df.filter(pl.col("conv_id").is_in(conv_ids))
        print(f"Filtered to {len(filtered):,} rows for sampled conversations")
        return filtered
    except Exception as e:
        print(f"Error loading parquet: {e}")
        sys.exit(1)


def build_conversation_texts(df: pl.DataFrame) -> dict:
    """
    Reconstruct full conversation text per conv_id.
    Concatenates all input + output turns in chronological order.
    """
    try:
        grouped = (
            df.sort(["conv_id", "date"])
            .group_by("conv_id", maintain_order=True)
            .agg([
                pl.col("input").str.concat("\n").alias("full_input"),
                pl.col("output").str.concat("\n").alias("full_output"),
                pl.len().alias("n_turns"),
            ])
        )

        conversations = {}
        for row in grouped.iter_rows(named=True):
            conv_id = row["conv_id"]
            full_text = row["full_input"] + "\n" + row["full_output"]
            conversations[conv_id] = {
                "text": full_text,
                "n_turns": row["n_turns"],
            }
        print(f"Built text for {len(conversations):,} conversations")
        return conversations
    except Exception as e:
        print(f"Error building conversation texts: {e}")
        sys.exit(1)


def tokenize_conversations(conversations: dict, tokenizer) -> list:
    """
    Tokenize all conversations and compute per-conversation token counts.
    """
    results = []
    for conv_id, data in conversations.items():
        try:
            tokens = tokenizer.encode(data["text"])
            results.append({
                "conv_id": conv_id,
                "n_turns": data["n_turns"],
                "total_tokens": len(tokens),
            })
        except Exception as e:
            print(f"Error tokenizing {conv_id}: {e}")
            results.append({
                "conv_id": conv_id,
                "n_turns": data["n_turns"],
                "total_tokens": 0,
            })

    print(f"Tokenized {len(results):,} conversations")
    return results


def compute_percentiles(results: list, windows: list) -> dict:
    """
    Compute P50 and coverage percentages for candidate window sizes.
    """
    tokens = [r["total_tokens"] for r in results]
    p50 = float(np.percentile(tokens, 50))
    max_tok = max(tokens)

    coverage = {}
    for window in windows:
        count = sum(1 for t in tokens if t <= window)
        pct = (count / len(tokens)) * 100
        coverage[window] = pct

    return {
        "p50": p50,
        "max": max_tok,
        "coverage": coverage,
        "tokens": tokens,
    }


def export_stats(results: list, output_path: str) -> None:
    """Export token stats to CSV."""
    try:
        df = pl.DataFrame(results)
        df.write_csv(output_path)
        print(f"Exported token stats to {output_path}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")


def print_report(stats: dict, windows: list) -> None:
    """Print analysis summary."""
    print("\n" + "=" * 50)
    print("TOKEN CONTEXT SIZE ANALYSIS")
    print("=" * 50)
    print(f"\nP50 tokens: {stats['p50']:.0f}")
    print(f"Max tokens: {stats['max']:.0f}")

    print(f"\nCoverage by window size:")
    for window in windows:
        pct = stats["coverage"][window]
        bar = "█" * int(pct / 5)
        print(f"  {window:>5}: {pct:5.1f}% {bar}")

    recommended = min(
        [w for w in windows if stats["coverage"][w] >= 95],
        default=windows[-1]
    )
    print(f"\nRecommended minimum window (95% coverage): {recommended}")


def main():
    tokenizer = load_tokenizer()
    conv_ids, _ = sample_conversation_ids(STATS_PATH, SAMPLE_FRACTION)
    df = load_and_filter_parquet(DATA_PATH, conv_ids)
    conversations = build_conversation_texts(df)
    results = tokenize_conversations(conversations, tokenizer)
    stats = compute_percentiles(results, WINDOWS)

    export_stats(results, os.path.join(OUTPUT_DIR, "token_stats.csv"))

    with open(os.path.join(OUTPUT_DIR, "percentiles.txt"), "w") as f:
        f.write(f"P50: {stats['p50']:.0f}\n")
        f.write(f"Max: {stats['max']:.0f}\n")
        f.write(f"\nCoverage:\n")
        for window in WINDOWS:
            f.write(f"  {window}: {stats['coverage'][window]:.1f}%\n")

    print_report(stats, WINDOWS)


if __name__ == "__main__":
    main()