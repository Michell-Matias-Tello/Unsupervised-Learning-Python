# Digital Content Consumption Patterns

**Customer Analytics Portfolio — Unsupervised Learning**

End-to-end pipeline that generates synthetic session-level data from a digital
content platform and discovers behavioral consumption segments ("digital
personas") using dimensionality reduction (PCA / t-SNE) and clustering
(K-Means, with DBSCAN diagnostics).

---

## Project Overview

The project simulates user sessions and identifies four latent behavioral
clusters:

| Persona             | Behavioral Signature                                                        |
|---------------------|-----------------------------------------------------------------------------|
| Marathon Finishers  | Long sessions, near-complete depth, high interactions, desktop, daytime     |
| Sampling Nomads     | Many pieces, very low depth, high diversity, mobile, afternoon/evening      |
| Burst Consumers     | Short intense sessions, moderate depth, mobile-only, sporadic timing        |
| Download & Depart   | Very brief sessions, high save/download actions, minimal in-session depth   |

---
## Project Structure

The project follows a modular and reproducible structure:

```plaintext
.
├── data/
│   ├── raw/                  # Synthetic dataset, data dictionary, cluster definitions
│   └── processed/            # Scaled matrix, PCA/t-SNE outputs, cluster assignments
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_clustering_modeling.ipynb
│   └── 03_evaluation_insights.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized paths, columns, and constants
│   ├── data_generation.py    # Phase 0: Synthetic data generation
│   ├── eda.py                # Phase 1: Exploratory analysis
│   ├── preprocessing.py      # Phase 2: Scaling and encoding
│   ├── dimensionality_reduction.py  # Phase 3: PCA and t-SNE
│   ├── clustering.py         # Phase 4: K-Means clustering
│   ├── profiling.py          # Phase 5: Cluster profiling and persona mapping
│   └── pipeline.py           # Orchestrates all phases
│
├── models/                  # Persisted models (e.g., .pkl files)
│
├── reports/
│   ├── figures/              # Visualizations: EDA, clustering diagnostics, and profiling
│   │   ├── univariate_distributions.png
│   │   ├── boxplot_outliers.png
│   │   ├── correlation_heatmap.png
│   │   ├── categorical_distributions.png
│   │   ├── pairplot_relationships.png
│   │   ├── preprocessing_scaling_impact.png
│   │   ├── pca_explained_variance.png
│   │   ├── pca_loadings_heatmap.png
│   │   ├── dual_projection_comparison.png
│   │   ├── kmeans_diagnostics.png
│   │   ├── dbscan_kdistance.png
│   │   ├── silhouette_validation.png
│   │   └── cluster_radar_profiles.png
│   │
│   └── tables/               # Summary tables: EDA, preprocessing, clustering, and profiling
│       ├── descriptive_statistics.csv
│       ├── outlier_analysis_summary.csv
│       ├── spearman_correlation_matrix.csv
│       ├── categorical_distributions.csv
│       ├── data_quality_report.csv
│       ├── scaling_report.csv
│       ├── onehot_encoding_map.csv
│       ├── final_feature_matrix_columns.csv
│       ├── pca_variance_breakdown.csv
│       ├── pca_dominant_features.csv
│       ├── kmeans_k_selection.csv
│       ├── kmeans_cluster_distribution.csv
│       ├── dbscan_grid_search.csv
│       ├── dbscan_failure_diagnosis.csv
│       ├── model_selection_comparison.csv
│       ├── silhouette_per_cluster.csv
│       ├── cluster_numerical_profiles.csv
│       ├── cluster_categorical_profiles.csv
│       ├── cluster_persona_mapping.csv
│       └── cluster_persona_detailed.csv
│
├── requirements.txt
└── README.md

````

## Installation


1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution

### Option 1: Full Pipeline
Run the entire pipeline in one command:
```bash
python -m src.pipeline
```
This executes all phases sequentially:
**Data Generation → EDA → Preprocessing → Dimensionality Reduction → Clustering → Profiling**.

### Option 2: Step-by-Step Execution
1. **Generate synthetic data:**
   ```bash
   python generate_project.py
   ```
   Outputs:
   - `src/` with all phase modules
   - `notebooks/` with Jupyter notebooks
   - `data/raw/` with the dataset and definitions

2. **Run individual phases:**
   Open and execute the notebooks in order:
   - `01_exploratory_analysis.ipynb`
   - `02_clustering_modeling.ipynb`
   - `03_evaluation_insights.ipynb`

---

## Pipeline Phases

| **Phase** | **Module**                     | **Output**                                      |
|-----------|--------------------------------|-------------------------------------------------|
| 0         | `data_generation.py`          | `data/raw/digital_content_sessions.csv`         |
| 1         | `eda.py`                       | `reports/outlier_analysis_summary.csv`, heatmap |
| 2         | `preprocessing.py`             | `data/processed/sessions_preprocessed.csv`     |
| 3         | `dimensionality_reduction.py` | `sessions_pca.csv`, `sessions_tsne.csv`, loadings|
| 4         | `clustering.py`                | `cluster_assignments.csv`, `models/kmeans_model.pkl` |
| 5         | `profiling.py`                 | `reports/cluster_persona_mapping.csv`           |

---

## Methodology

### Data Processing
- **Scaling:** `StandardScaler` for numerical features (zero-mean, unit-variance).
- **Encoding:** One-hot encoding for categorical features.

### Dimensionality Reduction
- **PCA:** Retains ≥80% of variance for clustering input.
- **t-SNE:** Used for visualization only (not for clustering).

### Clustering
- **K-Means:** Optimal `k` selected via silhouette score, validated with Davies–Bouldin index.
- **DBSCAN:** Included for diagnostic comparison.

### Reproducibility
- Global `RANDOM_STATE = 42` for consistent results.

---

## Notes
- All scripts are designed to read/write files locally. Execute in a **local environment** (terminal, VS Code, Jupyter).
- t-SNE requires `scikit-learn ≥ 1.2` due to the `max_iter` parameter.

---

## License
This project is intended for **portfolio and educational purposes**.
