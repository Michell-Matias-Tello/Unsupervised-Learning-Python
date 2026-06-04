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


## Installation

Requires **Python 3.9+**.

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
Usage
Step 1 — Bootstrap the project
Run the generator script (the file that builds the structure, writes the src/ package and notebooks, and creates the dataset):

python generate_project.py
This produces:

src/ populated with all phase modules
notebooks/ with notebooks 01–03
data/raw/ with the dataset, data dictionary, and cluster definitions
Step 2 — Run the full pipeline
python -m src.pipeline
This executes all phases in order: data generation → EDA → preprocessing → dimensionality reduction → clustering → profiling.

Alternative — Run interactively
Open the notebooks in order:

notebooks/01_exploratory_analysis.ipynb
notebooks/02_clustering_modeling.ipynb
notebooks/03_evaluation_insights.ipynb
Pipeline Phases

content_copy
Copy
Phase	Module	Output
0	data_generation.py	data/raw/digital_content_sessions.csv
1	eda.py	reports/outlier_analysis_summary.csv, heatmap
2	preprocessing.py	data/processed/sessions_preprocessed.csv
3	dimensionality_reduction.py	sessions_pca.csv, sessions_tsne.csv, loadings
4	clustering.py	cluster_assignments.csv, models/kmeans_model.pkl
5	profiling.py	reports/cluster_persona_mapping.csv
Methodology Notes
Scaling: StandardScaler (zero-mean, unit-variance) for numerical features.
Encoding: One-hot encoding for nominal categorical features.
Dimensionality reduction: PCA retaining ≥80% of variance; t-SNE for visualization only (not used as clustering input).
Model selection: Optimal k chosen by silhouette score; validated with the Davies–Bouldin index. DBSCAN is included as a diagnostic comparison.
Reproducibility: Global RANDOM_STATE = 42.
Notes
All scripts read/write files on disk; run them in a local environment (terminal, VS Code, or Jupyter), not in a cloud spreadsheet runtime.
t-SNE uses the max_iter parameter, requiring scikit-learn ≥ 1.2.
License
For portfolio and educational use.


---
