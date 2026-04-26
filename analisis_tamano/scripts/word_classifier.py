"""
Word Classifier
--------------
Extracts and counts words from the input side of chatbot conversations.
Words are sequences of consecutive characters >= 3 chars.
"""

import polars as pl
import re
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_50k_anonymized.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")


def load_data(path: str) -> pl.DataFrame:
    """Load parquet dataset, return only input column."""
    try:
        df = pl.read_parquet(path)
        print(f"Loaded {len(df):,} rows from {path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        raise
    except Exception as e:
        print(f"Error loading parquet: {e}")
        raise


def extract_words(text: str) -> list:
    """
    Extract words from text.
    - Split on whitespace
    - Only keep tokens with len > 2
    - Lowercase for case-insensitive counting
    - Strip punctuation from word boundaries
    """
    if not isinstance(text, str):
        return []

    tokens = text.split()
    words = []
    for token in tokens:
        cleaned = token.strip('.,;:!?()[]{}"\'¿?¡!«»""''--')
        if len(cleaned) > 2:
            words.append(cleaned.lower())
    return words


def process_all_inputs(df: pl.DataFrame) -> Counter:
    """
    Process all input rows and aggregate word frequencies.
    """
    word_counts = Counter()
    total_inputs = len(df)

    for idx, text in enumerate(df["input"]):
        words = extract_words(text)
        word_counts.update(words)

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1:,} / {total_inputs:,} rows...")

    print(f"Processed all {total_inputs:,} rows")
    return word_counts


def export_word_counts(word_counts: Counter, output_path: str) -> None:
    """
    Export word counts to CSV, sorted by frequency descending.
    """
    try:
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        df = pl.DataFrame(sorted_words, schema=["word", "count"])
        df.write_csv(output_path)
        print(f"Exported {len(df):,} unique words to {output_path}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        raise


def print_summary(word_counts: Counter, top_n: int = 20) -> None:
    """Print summary statistics and top words."""
    total_words = sum(word_counts.values())
    unique_words = len(word_counts)

    print(f"\n=== Word Frequency Summary ===")
    print(f"Total word occurrences: {total_words:,}")
    print(f"Unique words: {unique_words:,}")
    print(f"\nTop {top_n} most common words:")
    for word, count in word_counts.most_common(top_n):
        print(f"  {word:<20} {count:>6,}")


def main():
    df = load_data(DATA_PATH)
    word_counts = process_all_inputs(df)
    export_word_counts(word_counts, os.path.join(OUTPUT_DIR, "word_counts.csv"))
    print_summary(word_counts)


if __name__ == "__main__":
    main()