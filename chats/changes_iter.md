# Changelog - TF-IDF Clustering Iterations

## Iteration History

| Iteration | File | HDBSCAN Params | UMAP n_neighbors | Date | Clusters | Silhouette |
|-----------|------|---------------|-----------------|------|----------|------------|
| 1 (baseline) | `tfidf_clustering.py` | min_cluster_size=75, min_samples=10 | 25 | 2026-04-26 | 72 | 0.567 |
| 2 | `iteration_2.py` | min_cluster_size=150, min_samples=15 | 25 | 2026-04-26 | 36 | 0.528 |
| 3 | `iteration_3.py` | min_cluster_size=125, min_samples=15 | 25 | 2026-04-26 | 43 | 0.549 |
| 4 | `iteration_4.py` | min_cluster_size=125, min_samples=15 | **30** | 2026-04-26 | 43 | 0.544 |
| 5 | `iteration_5.py` | min_cluster_size=125, min_samples=15 | **20** | 2026-04-26 | 43 | 0.548 |
| 6 | `iteration_6.py` | min_cluster_size=**150**, min_samples=15 | **20** | 2026-04-26 | **35** | 0.545 |
| 7 | `iteration_7.py` | min_cluster_size=**100**, min_samples=15 | **20** | 2026-04-26 | **48** | 0.543 |
| 8 | `iteration_8.py` | min_cluster_size=100, min_samples=15 | **20** | 2026-04-26 | **55** | **0.598** |
| 9 | `iteration_9_stopwords.py` | min_cluster_size=100, min_samples=15 | **20** | 2026-04-26 | 54 | 0.591 |
| 10 | `iteration_10_top10.py` | min_cluster_size=100, min_samples=15 | **20** | 2026-04-26 | 55 | 0.585 |
| 11 | `iteration_11_notarjeta.py` | min_cluster_size=100, min_samples=15 | **20** | 2026-04-26 | **60** | **0.601** |

---

## Iteration 1 → Iteration 2

### HDBSCAN Parameters
- `min_cluster_size`: 75 → **150**
- `min_samples`: 10 → **15**

### Data Loading
- **Removed**: Fresh data loading from parquet
- **Added**: Load precomputed UMAP and TF-IDF from iteration 1
  ```python
  umap_5d = np.load(DATA_PATH / 'umap_5d.npy')
  tfidf_matrix = load_npz(DATA_PATH / 'tfidf_matrix.npz')
  first_turns = pl.read_parquet(DATA_PATH / 'first_turns_clustered.parquet')
  ```

### Output Handling
- **Added**: Output subdirectory structure
  ```python
  OUTPUT_PATH = DATA_PATH / 'outputs' / 'iteration_2_medium'
  OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
  ```
- **Changed**: All output files saved to `OUTPUT_PATH/` instead of `DATA_PATH/`

### Visualization
- **Changed**: `plt.show()` → `plt.close()` (non-interactive saving)
- **Updated**: Title reflects iteration parameters

### Summary Files
- **Added**: `summary.txt` with parameters and metrics
- **Added**: `cluster_details.txt` with per-cluster info
- **Added**: `cluster_details` list to store info for file output

### Keyword Extraction
- **Unchanged**: Uses `get_top_tfidf_terms` function

### Code Cleanup
- **Removed**: Import of `TfidfVectorizer` (not needed when loading precomputed)
- **Removed**: `import re` (not needed when loading precomputed)
- **Removed**: `import pickle` (not needed in iteration 2)
- **Removed**: `save_npz`, `pickle.dump` (artifacts already saved in iter 1)

---

## Iteration 2 → Iteration 3

### HDBSCAN Parameters
- `min_cluster_size`: 150 → **125**
- `min_samples`: 15 → **15** (unchanged)

### Data Loading
- **Changed**: Load from iteration 2 outputs instead of iteration 1
  ```python
  iter2_path = DATA_PATH / 'outputs' / 'iteration_2_medium'
  umap_5d = np.load(iter2_path / 'umap_5d.npy')
  first_turns = pl.read_parquet(iter2_path / 'first_turns_clustered.parquet')
  ```

### Output Directory
- **Changed**: `iteration_2_medium` → **`iteration_3_125`**

### Summary Files
- **Unchanged**: Same `summary.txt` and `cluster_details.txt` structure

---

## Iteration 3 Fix (2026-04-26)

### Issue
- Iteration 3 initially failed to load TF-IDF matrix (`tfidf_matrix.npz` not found)
- Keyword extraction was degraded to simple word frequency (not TF-IDF based)

### Resolution
- Created `regen_tfidf.py` to regenerate TF-IDF artifacts from saved parquet data
- Re-ran `regen_tfidf.py` successfully:
  - TF-IDF matrix: (24119, 5000)
  - Vocabulary: 5000 features
  - Saved to: `tfidf_matrix.npz`, `tfidf_vectorizer.pkl`
- Updated `iteration_3.py` to properly load TF-IDF from `DATA_PATH`

### Result
- Keywords now properly extracted using TF-IDF (bigrams like 'hola que tal', 'cargo no reconocido')

---

## Iteration 4 - n_neighbors=30 (2026-04-26)

### UMAP Parameters Changed
- `n_neighbors`: 25 → **30**
- `min_dist`: 0.1 (unchanged)
- `n_components`: 5 (unchanged)

### HDBSCAN Parameters
- Same as iteration 3: `min_cluster_size=125`, `min_samples=15`

### Key Finding
**Increasing n_neighbors increased noise significantly:**
- Iteration 3 (n_neighbors=25): 30.1% noise
- Iteration 4 (n_neighbors=30): **40.3% noise**

**Explanation:** More neighbors creates denser connectivity in the UMAP graph, making it harder for HDBSCAN to separate distinct clusters.

### Results
- Clusters: 43 (same as iter 3)
- Silhouette: 0.544 (slightly lower than iter 3's 0.549)
- Noise: 40.3% (higher than iter 3's 30.1%)
- Clustered: 59.7%

### Output Directory
- `outputs/iteration_4_neighbors30/`

---

## Iteration 5 - n_neighbors=20 (2026-04-26)

### UMAP Parameters Changed
- `n_neighbors`: 25 → **20**
- `min_dist`: 0.1 (unchanged)
- `n_components`: 5 (unchanged)

### HDBSCAN Parameters
- Same as iteration 3: `min_cluster_size=125`, `min_samples=15`

### Key Finding
**Decreasing n_neighbors slightly reduced noise:**
- Iteration 3 (n_neighbors=25): 30.1% noise, silhouette 0.549
- Iteration 5 (n_neighbors=20): **29.5% noise**, silhouette 0.548

**Explanation:** Fewer neighbors = sparser connectivity = HDBSCAN can better separate distinct clusters.

### Results
- Clusters: 43 (same as iter 3-4)
- Silhouette: 0.548 (similar to iter 3's 0.549)
- Noise: 29.5% (lower than iter 3's 30.1%)
- Clustered: 70.5%

### Output Directory
- `outputs/iteration_5_neighbors20/`

---

## Iteration 6 - n_neighbors=20, min_cluster_size=150 (2026-04-26)

### Parameters Changed
- `n_neighbors`: 25 → **20**
- `min_cluster_size`: 125 → **150**
- `min_samples`: 15 (unchanged)

### Key Finding
**Increasing min_cluster_size to 150 while keeping n_neighbors=20:**
- Iteration 5 (n=20, min_cluster=125): 43 clusters, 29.5% noise
- Iteration 6 (n=20, min_cluster=150): **35 clusters**, 32.4% noise

**Trade-off:** More aggressive clustering (larger min_cluster_size) reduces clusters but slightly increases noise.

### Results
- Clusters: 35 (reduced from 43)
- Silhouette: 0.545
- Noise: 32.4%
- Clustered: 67.6%

### Output Directory
- `outputs/iteration_6_150_20/`

---

## Iteration 7 - n_neighbors=20, min_cluster_size=100 (2026-04-26)

### Parameters Changed
- `n_neighbors`: 25 → **20**
- `min_cluster_size`: 125 → **100**
- `min_samples`: 15 (unchanged)

### Key Finding
**Decreasing min_cluster_size to 100 with n_neighbors=20:**
- Iteration 5 (n=20, min_cluster=125): 43 clusters, 29.5% noise
- Iteration 6 (n=20, min_cluster=150): 35 clusters, 32.4% noise
- Iteration 7 (n=20, min_cluster=100): **48 clusters**, **27.3% noise** ← best noise so far

**Trade-off:** More clusters with lower noise - best overall balance.

### Results
- Clusters: 48 (most so far)
- Silhouette: 0.543
- Noise: **27.3%** (lowest so far)
- Clustered: **72.7%** (highest so far)

### Notable New Clusters
- Cluster 12: portabilidad (104 terms)
- Cluster 15: tasa de interés (interest rate)
- Cluster 16: retiro sin tarjeta (cardless withdrawal)
- Cluster 24: notificaciones/transferencias programadas
- Cluster 44: meses sin intereses (months without interest)

### Output Directory
- `outputs/iteration_7_100_20/`

---

## Iteration 8 - min_dist=0.05 (2026-04-26)

### Parameters Changed
- `min_dist`: 0.1 → **0.05** (tighter clusters)
- `n_neighbors`: 20 (same as iter 7)
- `min_cluster_size`: 100 (same as iter 7)

### Key Finding
**Lower min_dist=0.05 significantly improved silhouette:**
- Iteration 7 (min_dist=0.1, n=20, min_cluster=100): 48 clusters, silhouette 0.543, 27.3% noise
- Iteration 8 (min_dist=0.05, n=20, min_cluster=100): **55 clusters**, **silhouette 0.598**, 26.8% noise

**Best iteration so far for silhouette!**

### Results
- Clusters: 55 (most so far)
- Silhouette: **0.598** (highest so far, beats baseline 0.567)
- Noise: 26.8% (tied for lowest)
- Clustered: 73.2% (highest so far)

### Notable Clusters
- Cluster 3: no thanks (asesor de aclaraciones) - new
- Cluster 20: cashback, anualidad, inversión - new
- Cluster 30: saldo retenido/garantía - new
- Cluster 38: puede transferir / se puede - new
- Cluster 40: crédito personal (solicitar) - new

### Output Directory
- `outputs/iteration_8_mindist05/`

---

## Iteration 9 - Stop Words (108 words) (2026-04-26)

### Parameters Changed
- Stop words: Custom list of 108 Spanish stop words + domain-specific
- UMAP: n_neighbors=20, min_dist=0.05 (same as iter 8)
- HDBSCAN: min_cluster_size=100, min_samples=15 (same as iter 8)

### Key Finding
**Aggressive stop word removal hurt clustering quality:**
- Iteration 8 (no stop words): 55 clusters, silhouette 0.598, 26.8% noise
- Iteration 9 (108 stop words): 54 clusters, **silhouette 0.591**, **31.0% noise**

**Trade-off:** Removed both noise AND useful distinguishing words.

### Results
- Clusters: 54
- Silhouette: 0.591 (lower than iter 8)
- Noise: 31.0% (higher than iter 8)
- Clustered: 69.0%

### Output Directory
- `outputs/iteration_9_stopwords/`

---

## Iteration 10 - TOP-10 Stop Words Only (2026-04-26)

### Parameters Changed
- Stop words: Only top-10 most frequent words (`de, mi, tarjeta, no, la, me, que, puedo, el, en`)
- UMAP: n_neighbors=20, min_dist=0.05 (same as iter 8)
- HDBSCAN: min_cluster_size=100, min_samples=15 (same as iter 8)

### Key Finding
**Selective stop word removal improved noise significantly:**
- Iteration 8 (no stop words): 55 clusters, silhouette 0.598, 26.8% noise, 73.2% clustered
- Iteration 10 (top-10 stop words): 55 clusters, silhouette 0.585, **21.3% noise**, **78.7% clustered**

**Trade-off:** Lower silhouette (0.585 vs 0.598) but much better noise reduction and clustering coverage.

### Results
- Clusters: 55
- Silhouette: 0.585
- Noise: **21.3%** ← lowest noise
- Clustered: **78.7%** ← highest clustered

### Notable Clusters
- Cluster 0: hola (greeting) - now separated
- Cluster 1: cancelar/gracias (cancellation)
- Cluster 10: apple pay wallet (cleaner digital wallet)
- Cluster 12: hablar con asesor (human routing)

### Output Directory
- `outputs/iteration_10_top10/`

---

## Iteration 11 - TOP-10 Stop Words (tarjeta EXCLUDED) (2026-04-26)

### Parameters Changed
- Stop words: Top 10 (tarjeta EXCLUDED - domain relevant)
  - Removed: `de, el, en, la, me, mi, no, puedo, que, un`
  - KEPT: `tarjeta` (domain relevant word)
- UMAP: n_neighbors=20, min_dist=0.05 (same as iter 8)
- HDBSCAN: min_cluster_size=100, min_samples=15 (same as iter 8)

### Key Finding
**Including "tarjeta" improved silhouette significantly:**
- Iteration 10 (tarjeta removed): 55 clusters, silhouette 0.585, 21.3% noise, 78.7% clustered
- Iteration 11 (tarjeta kept): **60 clusters**, **silhouette 0.601**, 30.6% noise, 69.4% clustered

**Best silhouette so far!** More clusters with better semantic separation.

### Results
- Clusters: **60** ← most clusters
- Silhouette: **0.601** ← best silhouette
- Noise: 30.6%
- Clustered: 69.4%

### Notable Clusters
- Cluster 44 (3603): still large but "tarjeta" now distinguishes it
- Cluster 55: tarjeta física (physical card)
- Cluster 54: tarjeta crédito / anualidad
- Cluster 53: deuda tarjeta
- Cluster 39: tarjeta virtual

### Output Directory
- `outputs/iteration_11_notarjeta/`

---

## Summary of Parameter Impact

| Metric | Iter 8 (baseline) | Iter 10 (tarjeta rm) | Iter 11 (tarjeta keep) |
|--------|-------------------|----------------------|------------------------|
| Clusters | 55 | 55 | **60** ← most |
| Silhouette | 0.598 | 0.585 | **0.601** ← best |
| Noise % | 26.8% | **21.3%** ← best | 30.6% |
| Clustered % | 73.2% | **78.7%** ← best | 69.4% |

### Best Metrics by Category:
- **Silhouette**: **Iteration 11 (0.601)** ← NEW BEST
- **Noise %**: Iteration 10 (21.3%)
- **Clustered %**: Iteration 10 (78.7%)
- **Most Clusters**: **Iteration 11 (60)**

---

## Output Files Per Iteration

### Iteration 1 (baseline)
```
tfidf_clustering.py
tfidf_matrix.npz
tfidf_vectorizer.pkl
umap_5d.npy
cluster_labels.npy
first_turns_clustered.parquet
cluster_distribution.png
umap_clusters.png
```

### Iteration 2 (`outputs/iteration_2_medium/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 3 (`outputs/iteration_3_125/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 4 (`outputs/iteration_4_neighbors30/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 5 (`outputs/iteration_5_neighbors20/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 6 (`outputs/iteration_6_150_20/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 7 (`outputs/iteration_7_100_20/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 8 (`outputs/iteration_8_mindist05/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 9 (`outputs/iteration_9_stopwords/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
stopwords.txt
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 10 (`outputs/iteration_10_top10/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
stopwords_top10.txt
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

### Iteration 11 (`outputs/iteration_11_notarjeta/`)
```
first_turns_clustered.parquet
cluster_labels.npy
umap_5d.npy
tfidf_matrix.npz
tfidf_vectorizer.pkl
stopwords_top10_notarjeta.txt
cluster_distribution.png
umap_clusters.png
summary.txt
cluster_details.txt
```

---

## Notes

- Iteration 1 produces TF-IDF artifacts that iterations 2-3 reuse for UMAP
- Iterations 2-3 only re-run HDBSCAN clustering (not full pipeline)
- All iterations include 2D UMAP visualization

---

## TODO

- [x] iteration_3.py TF-IDF loading issue - **RESOLVED**
- [x] iteration_4 n_neighbors=30 test - **COMPLETED** (increases noise)
- [x] iteration_5 n_neighbors=20 test - **COMPLETED** (reduces noise to 29.5%)
- [x] iteration_6 min_cluster_size=150 + n_neighbors=20 - **COMPLETED** (35 clusters, 32.4% noise)
- [x] iteration_7 min_cluster_size=100 + n_neighbors=20 - **COMPLETED** (48 clusters, 27.3% noise)
- [x] iteration_8 min_dist=0.05 + iter7 params - **COMPLETED** (55 clusters, 0.598 silhouette)
- [x] iteration_9 108 stop words - **COMPLETED** (worse metrics - removed too many words)
- [x] iteration_10 top-10 stop words - **COMPLETED** (21.3% noise, 78.7% clustered - best for noise!)
- [x] iteration_11 top-10 stop words (tarjeta keep) - **COMPLETED** (**60 clusters, 0.601 silhouette** - NEW BEST!)
- [ ] Test min_dist=0.02 (lower) for potential further improvement
- [ ] Category mapping for iteration 11
- [ ] Compare iterations with stability index
- [ ] Compare iterations 8 and 10 with category mapping