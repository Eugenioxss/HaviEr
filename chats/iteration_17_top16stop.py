"""
TF-IDF + UMAP Clustering - Iteration 17 (Top-16 Generic Stopwords)

Goal: Branch from Iter 15, adding 'tengo' as stopword (16 total)

Stopwords removed: de, mi, no, la, me, que, el, en, un, a, como, y, para, con, si, tengo
Stopwords KEPT: tarjeta, credito, cuenta + all other banking terms

Parameters:
- TF-IDF: stop_words=16 (16 generic words, no banking terms)
- UMAP: n_neighbors=20, min_dist=0.05, n_components=5, metric='cosine'
- HDBSCAN: min_cluster_size=100, min_samples=15 (same as Iter 11)
"""

import polars as pl
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import umap
import hdbscan
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz
import pickle
import re

plt.style.use('seaborn-v0_8-whitegrid')
DATA_PATH = Path('/home/diego/gits/datathon_master/HaviEr/chats')
OUTPUT_PATH = DATA_PATH / 'outputs' / 'iteration_17_top16stop'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ITERATION 17 - TOP 16 GENERIC STOPWORDS (+ tengo)")
print("=" * 60)

STOP_WORDS = ['de', 'mi', 'no', 'la', 'me', 'que', 'el', 'en', 'un', 'a',
              'como', 'y', 'para', 'con', 'si', 'tengo']


# =============================================================================
# 1. Load data
# =============================================================================
print("\nLoading data...")
df = pl.read_parquet(DATA_PATH / 'dataset_50k_anonymized.parquet')
print(f"Total rows: {len(df)}")


# =============================================================================
# 2. Extract first-turn interactions
# =============================================================================
print("\nExtracting first-turn interactions...")

df_sorted = df.sort(['conv_id', 'date'])
first_turns = df_sorted.group_by('conv_id', maintain_order=True).first()
print(f"First-turn interactions: {len(first_turns)}")


# =============================================================================
# 3. Text preprocessing (standard)
# =============================================================================
print("\nPreprocessing text...")

def preprocess_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\sáéíóúüñ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

first_turns = first_turns.with_columns([
    pl.col('input').map_elements(preprocess_text, return_dtype=pl.String).alias('input_clean')
])
texts = first_turns['input_clean'].to_list()

print(f"Sample (processed): {texts[0][:80]}...")
print(f"Stopwords removed: {len(STOP_WORDS)} words")
print(f"Stopwords kept: tarjeta, credito, cuenta (banking terms)")


# =============================================================================
# 4. TF-IDF vectorization (with stop_words)
# =============================================================================
print("\nComputing TF-IDF vectors (with 16 stopwords)...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    sublinear_tf=True,
    stop_words=STOP_WORDS
)
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"TF-IDF shape: {tfidf_matrix.shape}")

feature_names = vectorizer.get_feature_names_out()
print(f"Total features: {len(feature_names)}")


# =============================================================================
# 5. Verify stopwords filtering
# =============================================================================
tarjeta_in_vocab = 'tarjeta' in feature_names
credito_in_vocab = 'crédito' in feature_names or 'credito' in feature_names
cuenta_in_vocab = 'cuenta' in feature_names
print(f"\nVerifying key terms in vocabulary:")
print(f"  'tarjeta' present: {tarjeta_in_vocab}")
print(f"  'crédito'/'credito' present: {credito_in_vocab}")
print(f"  'cuenta' present: {cuenta_in_vocab}")


# =============================================================================
# 6. Add channel_source feature
# =============================================================================
print("\nAdding channel_source feature...")

channel_feature = first_turns['channel_source'].cast(pl.Int32).to_numpy().reshape(-1, 1)
channel_feature = channel_feature / channel_feature.max()
X = hstack([tfidf_matrix, channel_feature])
print(f"Combined feature shape: {X.shape}")


# =============================================================================
# 7. UMAP dimensionality reduction (n_neighbors=20, min_dist=0.05)
# =============================================================================
print("\nRunning UMAP (n_neighbors=20, min_dist=0.05)...")

reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.05,
    n_components=5,
    metric='cosine',
    random_state=42
)
umap_5d = reducer.fit_transform(X.toarray())
print(f"UMAP result shape: {umap_5d.shape}")


# =============================================================================
# 8. HDBSCAN clustering (min_cluster_size=100, min_samples=15)
# =============================================================================
print("\nRunning HDBSCAN (min_cluster_size=100, min_samples=15)...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=15,
    metric='euclidean'
)
cluster_labels = clusterer.fit_predict(umap_5d)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")


# =============================================================================
# 9. Cluster distribution visualization
# =============================================================================
print("\nVisualizing cluster distribution...")

cluster_counts = Counter(cluster_labels)
if -1 in cluster_counts:
    del cluster_counts[-1]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
clusters = sorted(cluster_counts.keys())
counts = [cluster_counts[c] for c in clusters]
plt.bar(clusters, counts, color='steelblue')
plt.xlabel('Cluster ID')
plt.ylabel('Count')
plt.title('Cluster Size Distribution\n(UMAP n_neighbors=20, min_dist=0.05, HDBSCAN 100/15)\nTop-16 generic stopwords removed (+ tengo)')

plt.subplot(1, 2, 2)
plt.pie(counts, labels=[f'C{c}' for c in clusters], autopct='%1.1f%%')
plt.title('Cluster Proportions')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'cluster_distribution.png', dpi=150)
plt.close()

print(f"Cluster sizes: {dict(sorted(cluster_counts.items()))}")


# =============================================================================
# 10. 2D UMAP visualization
# =============================================================================
print("\nComputing 2D UMAP for visualization...")

reducer_2d = umap.UMAP(
    n_neighbors=20,
    min_dist=0.05,
    n_components=2,
    metric='cosine',
    random_state=42
).fit(X.toarray())
umap_2d = reducer_2d.embedding_

plt.figure(figsize=(14, 10))
scatter = plt.scatter(umap_2d[:, 0], umap_2d[:, 1],
                      c=cluster_labels, cmap='tab20',
                      s=5, alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Projection - Iteration 17 (Top-16 Generic Stopwords)')
plt.savefig(OUTPUT_PATH / 'umap_clusters.png', dpi=150)
plt.close()


# =============================================================================
# 11. Keyword extraction per cluster
# =============================================================================
print("\nExtracting keywords per cluster...")

def get_top_tfidf_terms(tfidf_matrix, cluster_indices, vectorizer, n=10):
    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
    top_indices = cluster_tfidf.argsort()[-n:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    return [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

result_df = first_turns.with_columns([
    pl.Series('cluster', cluster_labels)
])

cluster_details = []
print("\n=== Cluster Analysis ===\n")
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue
    mask = np.array(cluster_labels) == cluster_id
    top_terms = get_top_tfidf_terms(tfidf_matrix, mask, vectorizer, n=10)
    sample_texts = [texts[i] for i in np.where(mask)[0][:2]]

    info = {
        'cluster_id': int(cluster_id),
        'size': int(mask.sum()),
        'top_terms': [t[0] for t in top_terms],
        'sample': sample_texts[0][:80] if sample_texts else ''
    }
    cluster_details.append(info)

    print(f"Cluster {cluster_id} (n={mask.sum()}):")
    print(f"  Top terms: {[t[0] for t in top_terms]}")
    print(f"  Sample: {sample_texts[0][:80]}...\n")


# =============================================================================
# 12. Quality metrics
# =============================================================================
print("\nComputing quality metrics...")

mask = cluster_labels != -1
silhouette = silhouette_score(umap_5d[mask], cluster_labels[mask])
print(f"Silhouette Score (excluding noise): {silhouette:.3f}")

total = len(cluster_labels)
clustered = mask.sum()
print(f"Clustered: {clustered}/{total} ({100*clustered/total:.1f}%)")
print(f"Noise: {total - clustered} ({100*(total-clustered)/total:.1f}%)")


# =============================================================================
# 13. Save results
# =============================================================================
print("\nSaving results...")

result_df.write_parquet(OUTPUT_PATH / 'first_turns_clustered.parquet')
np.save(OUTPUT_PATH / 'umap_5d.npy', umap_5d)
np.save(OUTPUT_PATH / 'cluster_labels.npy', cluster_labels)
save_npz(OUTPUT_PATH / 'tfidf_matrix.npz', tfidf_matrix)

with open(OUTPUT_PATH / 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saved to:", OUTPUT_PATH)


# =============================================================================
# 14. Write summary files
# =============================================================================
with open(OUTPUT_PATH / 'summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("ITERATION 17 - TOP 16 GENERIC STOPWORDS (+ tengo)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Stopwords removed: " + str(STOP_WORDS) + "\n")
    f.write("Stopwords kept: ['tarjeta', 'credito', 'cuenta'] + all banking terms\n\n")
    f.write("Parameters:\n")
    f.write("  - TF-IDF max_features: 5000\n")
    f.write("  - TF-IDF ngram_range: (1, 2)\n")
    f.write("  - TF-IDF stop_words: 16 generic words\n")
    f.write("  - UMAP n_neighbors: 20\n")
    f.write("  - UMAP min_dist: 0.05\n")
    f.write("  - HDBSCAN min_cluster_size: 100\n")
    f.write("  - HDBSCAN min_samples: 15\n\n")
    f.write(f"Clusters found: {n_clusters}\n")
    f.write(f"Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)\n")
    f.write(f"Silhouette score: {silhouette:.3f}\n")
    f.write(f"Clustered: {clustered}/{total} ({100*clustered/total:.1f}%)\n\n")
    f.write("Cluster sizes:\n")
    for c in sorted(cluster_counts.keys()):
        f.write(f"  Cluster {c}: {cluster_counts[c]} ({100*cluster_counts[c]/len(cluster_labels):.1f}%)\n")

with open(OUTPUT_PATH / 'cluster_details.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CLUSTER DETAILS - ITERATION 17 (Top-16 Generic Stopwords)\n")
    f.write("=" * 60 + "\n\n")
    for info in cluster_details:
        f.write(f"Cluster {info['cluster_id']} (n={info['size']}):\n")
        f.write(f"  Top terms: {info['top_terms']}\n")
        f.write(f"  Sample: {info['sample']}\n\n")


# =============================================================================
# 15. Comparison summary
# =============================================================================
print("\n" + "=" * 60)
print("COMPARISON ACROSS ITERATIONS")
print("=" * 60)
print(f"{'Metric':<25} {'Iter 11':<12} {'Iter 15 (15)':<14} {'Iter 17 (16)':<14}")
print("-" * 70)
print(f"{'Clusters':<25} {'60':<12} {'60':<14} {n_clusters:<14}")
print(f"{'Silhouette':<25} {'0.601':<12} {'0.603':<14} {silhouette:<14.3f}")
print(f"{'Noise %':<25} {'30.6%':<12} {'22.3%':<14} {100*n_noise/len(cluster_labels):<14.1f}%")
print(f"{'Clustered %':<25} {'69.4%':<12} {'77.7%':<14} {100*clustered/total:<14.1f}%")


# =============================================================================
# 16. Final summary
# =============================================================================
print("\n" + "=" * 60)
print("=== ANALYSIS COMPLETE ===\n")
print(f"Output directory: {OUTPUT_PATH}")
print(f"Clusters: {n_clusters}")
print(f"Silhouette: {silhouette:.3f}")
print(f"Noise: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")