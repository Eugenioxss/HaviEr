"""
Apply category mapping to Iteration 17 clusters and create visualizations.
"""

import polars as pl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

DATA_PATH = Path('/home/diego/gits/datathon_master/HaviEr/chats')
OUTPUT_PATH = DATA_PATH / 'outputs' / 'iteration_17_top16stop' / 'category_mapping'

print("Loading cluster results...")
df = pl.read_parquet(OUTPUT_PATH.parent / 'first_turns_clustered.parquet')
print(f"Total rows: {len(df)}")

print("\nLoading mapping CSV...")
mapping_df = pl.read_csv(OUTPUT_PATH / 'mapping.csv')
print(f"Mapped clusters: {len(mapping_df)}")

mapping_dict = dict(zip(mapping_df['cluster_id'], mapping_df['category_id']))
category_names = dict(zip(mapping_df['category_id'], mapping_df['category_name']))

print("\nApplying category mapping...")
df = df.with_columns([
    pl.col('cluster').map_elements(lambda x: mapping_dict.get(x, -1), return_dtype=pl.Int32).alias('category_id')
])

category_counts = Counter(df['category_id'].to_list())
noise_count = category_counts.get(-1, 0)
del category_counts[-1]

print(f"\nCategory distribution (excluding noise {noise_count}):")
for cat_id in sorted(category_counts.keys()):
    cat_name = category_names.get(cat_id, 'Unknown')
    count = category_counts[cat_id]
    pct = 100 * count / (len(df) - noise_count)
    print(f"  {cat_id:2d}. {cat_name:<25} {count:5d} ({pct:5.1f}%)")

print(f"\nNoise (unmapped): {noise_count} ({100*noise_count/len(df):.1f}%)")

print("\nSaving categorized DataFrame...")
df.write_parquet(OUTPUT_PATH / 'first_turns_categorized.parquet')

cat_summary = df.group_by('category_id').agg([
    pl.len().alias('count'),
    pl.col('conv_id').count().alias('conversations')
]).sort('category_id')

cat_summary = cat_summary.with_columns([
    pl.col('category_id').map_elements(lambda x: category_names.get(x, 'Noise'), return_dtype=pl.String).alias('category_name')
])

cat_summary = cat_summary.sort('count', descending=True)
cat_summary.write_csv(OUTPUT_PATH / 'category_summary.csv')

print("Saved category_summary.csv")

print("\nCreating visualizations...")

plt.figure(figsize=(16, 10))

categories_sorted = sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True)
counts_sorted = [category_counts[c] for c in categories_sorted]
names_sorted = [category_names[c] for c in categories_sorted]
colors = plt.cm.Set3(range(len(categories_sorted)))

plt.subplot(1, 2, 1)
bars = plt.barh(range(len(categories_sorted)), counts_sorted, color=colors)
plt.yticks(range(len(categories_sorted)), names_sorted)
plt.xlabel('Conversations')
plt.title('Category Distribution\nIteration 17 - Top 16 Stopwords')
plt.gca().invert_yaxis()

for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
    pct = 100 * count / sum(counts_sorted)
    plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
             f'{count:,} ({pct:.1f}%)', va='center', fontsize=8)

plt.subplot(1, 2, 2)
wedges, texts, autotexts = plt.pie(counts_sorted, labels=names_sorted, autopct='%1.1f%%', pctdistance=0.75)
plt.setp(texts, fontsize=8)
plt.setp(autotexts, fontsize=7)
plt.title('Category Proportions')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'category_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved category_distribution.png")

print("\n" + "=" * 60)
print("CATEGORY MAPPING COMPLETE")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {OUTPUT_PATH / 'mapping.csv'}")
print(f"  - {OUTPUT_PATH / 'category_summary.csv'}")
print(f"  - {OUTPUT_PATH / 'first_turns_categorized.parquet'}")
print(f"  - {OUTPUT_PATH / 'category_distribution.png'}")
print(f"\nTotal categories: {len(category_counts)}")
print(f"Total conversations: {len(df)}")
print(f"Categorized: {len(df) - noise_count} ({100*(len(df) - noise_count)/len(df):.1f}%)")
print(f"Noise: {noise_count} ({100*noise_count/len(df):.1f}%)")