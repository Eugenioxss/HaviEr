# Solution structure
> intermadiate iterations and their outputs are excluded from this repo
- README.md -> Context for data interpretation
- Setup.md -> General agent instructiosn
- reqs_UMAP_analysis.md -> General solution instructions
- agent-session-history.md -> prompts and implementations made by agent
- tfidf_matrix.npz & tfidf_vectorizer-pkl -> analysis support data for different script iterations.
- first_turns_clustered.parquet -> filtered version of whole dataset to focus on initual queries
- changes_iter.md -> shows changes in iteration results
- outputs/
  - stopwords_top50.txt -> top 50 most generic words to be excluded from analysis
  - execution_results/ -> main results folder
    - category_mapping -> manual label assignment for result interpretation, includes resulting data, sample visualizations, and corresponding scripts.
