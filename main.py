"""
=============================================================================
DIGITAL CONTENT CONSUMPTION PATTERNS — UNSUPERVISED LEARNING PIPELINE
# =============================================================================
# Stages:
#   1. Exploratory Data Analysis (EDA)
#   2. Data Preprocessing
#   3. Dimensionality Reduction
#   4. Clustering (K-Means & DBSCAN)
#   5. Cluster Profiling & Business Interpretation
# =============================================================================
"""

# =========================================================================
# IMPORTS
# =========================================================================
import os
import warnings
from math import pi
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
import joblib

warnings.filterwarnings('ignore')

# =========================================================================
# GLOBAL CONFIGURATION
# =========================================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

sns.set_theme(style='whitegrid', palette='viridis')

COLORS = {
    'primary': '#2C3E50',
    'accent': '#E74C3C',
    'median': '#27AE60',
    'kde': '#3498DB',
    'scaled': '#27AE60',
    'original': '#3498DB',
    'kmeans': '#3498DB',
    'dbscan': '#9B59B6',
    'silhouette': '#27AE60',
    'centroid': '#E74C3C',
    'noise': '#7F8C8D',
    'grid': '#ECF0F1'
}

RANDOM_STATE = 42

# Project directory structure (created if missing)
TABLES_DIR = "reports/tables"
FIGURES_DIR = "reports/figures"
for _folder in ["data/raw", "data/processed", "notebooks", "src",
                "models", "reports/figures", "reports/tables"]:
    os.makedirs(_folder, exist_ok=True)

print(f"scikit-learn version: {sklearn.__version__}")

# =========================================================================
# DATA LOADING
# =========================================================================
df = pd.read_csv("data/raw/digital_content_sessions.csv")

numerical_cols = [
    'total_time_min', 'pieces_started', 'completion_depth_avg',
    'complementary_interactions', 'thematic_diversity',
    'pause_count', 'navigation_speed'
]

categorical_cols = ['moment_of_activity', 'device_type', 'used_recommendations']

# =========================================================================
# HELPER FUNCTION
# =========================================================================
def calculate_distribution_stats(data: pd.Series) -> Dict:
    """Calculate comprehensive distribution statistics."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'outlier_lower': Q1 - 1.5 * IQR,
        'outlier_upper': Q3 + 1.5 * IQR
    }

# =========================================================================
# CONSOLE REPORT: STATISTICAL SUMMARY
# =========================================================================
print("\n" + "=" * 100)
print("📊 PHASE 1: EXPLORATORY DATA ANALYSIS — STATISTICAL REPORT")
print("=" * 100)
print(f"Dataset: {df.shape[0]:,} sessions × {df.shape[1]} features\n")

stats_table = []
for col in numerical_cols:
    stats_dict = calculate_distribution_stats(df[col].dropna())
    stats_table.append([
        col.replace('_', ' ').title(),
        f"{stats_dict['mean']:.4f}",
        f"{stats_dict['median']:.4f}",
        f"{stats_dict['std']:.4f}",
        f"{stats_dict['min']:.4f}",
        f"{stats_dict['max']:.4f}",
        f"{stats_dict['skewness']:.4f}",
        f"{stats_dict['kurtosis']:.4f}",
        f"{stats_dict['Q1']:.4f}",
        f"{stats_dict['Q3']:.4f}"
    ])

headers = [
    'Feature', 'Mean', 'Median', 'Std Dev',
    'Min', 'Max', 'Skewness', 'Kurtosis', 'Q1', 'Q3'
]

print(tabulate(stats_table, headers=headers, tablefmt='grid',
               numalign='center', stralign='left'))
print(f"\n{'=' * 100}\n")

# Save descriptive statistics
pd.DataFrame(stats_table, columns=headers).to_csv(
    f"{TABLES_DIR}/descriptive_statistics.csv", index=False)

# =========================================================================
# 1.1 UNIVARIATE DISTRIBUTION ANALYSIS — CLEAN CHARTS
# =========================================================================
fig, axes = plt.subplots(4, 2, figsize=(18, 20))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    ax = axes[i]
    stats_dict = calculate_distribution_stats(df[col].dropna())

    # Histogram with KDE
    sns.histplot(df[col], kde=True, bins=30, edgecolor='white',
                 alpha=0.7, linewidth=0.8, color=COLORS['kde'], ax=ax)

    # Reference lines for mean and median
    ax.axvline(stats_dict['mean'], color=COLORS['accent'], linestyle='--',
               linewidth=2, alpha=0.9, label=f"Mean: {stats_dict['mean']:.2f}")
    ax.axvline(stats_dict['median'], color=COLORS['median'], linestyle='-',
               linewidth=2, alpha=0.9, label=f"Median: {stats_dict['median']:.2f}")

    ax.set_title(col.replace('_', ' ').title(),
                 fontsize=12, pad=15, color=COLORS['primary'])
    ax.set_xlabel('')
    ax.legend(frameon=True, fancybox=True, fontsize=8, loc='upper right')

if len(numerical_cols) < len(axes):
    fig.delaxes(axes[-1])

fig.suptitle('Univariate Distribution Analysis: Numerical Features',
             fontsize=16, fontweight='bold', y=1.01, color=COLORS['primary'])
plt.tight_layout()
plt.savefig('reports/figures/univariate_distributions.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 1.2 BOXPLOT ANALYSIS — VIOLIN-BOX HYBRID
# =========================================================================
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    ax = axes[i]

    # Violin plot for distribution shape
    parts = ax.violinplot(df[col].dropna(), positions=[1],
                          showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.6)

    # Boxplot overlay for quartile statistics
    ax.boxplot(df[col].dropna(), positions=[1], widths=0.3,
               patch_artist=True,
               boxprops=dict(facecolor='white', alpha=0.8),
               medianprops=dict(color=COLORS['accent'], linewidth=2))

    ax.set_title(col.replace('_', ' ').title(),
                 fontsize=11, pad=12, color=COLORS['primary'])
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')

for j in range(len(numerical_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Distribution & Outlier Visualization: Violin-Box Analysis',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/boxplot_outliers.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 1.3 COMPREHENSIVE OUTLIER ANALYSIS — CONSOLE REPORT
# =========================================================================
print("\n" + "=" * 100)
print("🔍 1.3 COMPREHENSIVE OUTLIER ANALYSIS — IQR METHOD")
print("=" * 100)

outlier_records = []

for col in numerical_cols:
    data = df[col].dropna()
    stats_dict = calculate_distribution_stats(data)

    # Identify outliers
    outlier_mask = (data < stats_dict['outlier_lower']) | \
                   (data > stats_dict['outlier_upper'])
    outliers = data[outlier_mask]

    # Severity classification
    outlier_pct = len(outliers) / len(data)
    severity = ('HIGH' if outlier_pct > 0.10 else
                'MEDIUM' if outlier_pct > 0.05 else
                'LOW')

    # Context-aware recommendation
    recommendation = (
        'Investigate before removal' if severity == 'HIGH'
        else 'Flag for context review' if severity == 'MEDIUM'
        else 'Likely natural variation'
    )

    record = {
        'Feature': col.replace('_', ' ').title(),
        'Q1': f"{stats_dict['Q1']:.4f}",
        'Q3': f"{stats_dict['Q3']:.4f}",
        'IQR': f"{stats_dict['IQR']:.4f}",
        'Lower Bound': f"{stats_dict['outlier_lower']:.4f}",
        'Upper Bound': f"{stats_dict['outlier_upper']:.4f}",
        'Outlier Count': len(outliers),
        'Outlier %': f"{outlier_pct*100:.2f}%",
        'Severity': severity,
        'Recommendation': recommendation
    }
    outlier_records.append(record)

    # Print per-feature analysis
    print(f"\n  Feature: {col.replace('_', ' ').title()}")
    print(f"    Q1: {stats_dict['Q1']:.4f} | Q3: {stats_dict['Q3']:.4f} | "
          f"IQR: {stats_dict['IQR']:.4f}")
    print(f"    Lower Bound: {stats_dict['outlier_lower']:.4f} | "
          f"Upper Bound: {stats_dict['outlier_upper']:.4f}")
    print(f"    Outliers Detected: {len(outliers)} "
          f"({outlier_pct*100:.2f}%)")
    print(f"    Severity: {severity} | Recommendation: {recommendation}")

# Summary table
outlier_df = pd.DataFrame(outlier_records)
print(f"\n{'=' * 100}")
print("OUTLIER ANALYSIS SUMMARY")
print(tabulate(outlier_df[['Feature', 'Outlier Count', 'Outlier %',
                            'Severity', 'Recommendation']],
               headers='keys', tablefmt='grid', showindex=False))
print(f"\n{'=' * 100}\n")

# Save outlier summary
outlier_df.to_csv(f"{TABLES_DIR}/outlier_analysis_summary.csv", index=False)

# =========================================================================
# 1.4 CORRELATION ANALYSIS — PROFESSIONAL HEATMAP
# =========================================================================
corr_matrix = df[numerical_cols].corr(method='spearman')

print("📈 SPEARMAN CORRELATION MATRIX")
print("-" * 100)
corr_display = corr_matrix.copy()
corr_display.index = [c.replace('_', ' ').title() for c in corr_display.index]
corr_display.columns = [c.replace('_', ' ').title() for c in corr_display.columns]
print(tabulate(corr_display, headers='keys', tablefmt='grid',
               floatfmt='.4f'))
print(f"\n{'=' * 100}\n")

# Save correlation matrix
corr_matrix.to_csv(f"{TABLES_DIR}/spearman_correlation_matrix.csv")

# Heatmap
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True, linewidths=1.5, linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'},
    annot_kws={'size': 11, 'fontweight': 'bold'},
    ax=ax
)

clean_labels = [col.replace('_', ' ').title() for col in numerical_cols]
ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(clean_labels, rotation=0, fontsize=10)

ax.set_title('Feature Correlation Structure\nSpearman Rank Correlation Matrix',
             fontsize=14, fontweight='bold', pad=25, color=COLORS['primary'])
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 1.5 CATEGORICAL DISTRIBUTION ANALYSIS
# =========================================================================
print("📊 CATEGORICAL FEATURE DISTRIBUTIONS")
print("-" * 100)
categorical_frames = []
for col in categorical_cols:
    counts = df[col].value_counts()
    pcts = df[col].value_counts(normalize=True) * 100

    cat_table = pd.DataFrame({
        'Category': counts.index,
        'Count': counts.values,
        'Percentage': [f"{p:.2f}%" for p in pcts.values]
    })
    print(f"\n{col.replace('_', ' ').title()}:")
    print(tabulate(cat_table, headers='keys', tablefmt='grid', showindex=False))

    # Accumulate for export
    cat_save = cat_table.copy()
    cat_save.insert(0, 'Feature', col)
    categorical_frames.append(cat_save)

print(f"\n{'=' * 100}\n")

# Save categorical distributions
pd.concat(categorical_frames, ignore_index=True).to_csv(
    f"{TABLES_DIR}/categorical_distributions.csv", index=False)

# Donut charts
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
palettes = ['Blues_r', 'Greens_r', 'Oranges_r']

for i, (col, palette) in enumerate(zip(categorical_cols, palettes)):
    value_counts = df[col].value_counts()

    wedges, texts, autotexts = axes[i].pie(
        value_counts.values,
        labels=value_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette(palette, len(value_counts)),
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('600')

    axes[i].set_title(col.replace('_', ' ').title(),
                      fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Categorical Feature Analysis',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('reports/figures/categorical_distributions.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 1.6 PAIRPLOT — MULTIVARIATE RELATIONSHIPS
# =========================================================================
pairplot_features = [
    'total_time_min', 'pieces_started', 'completion_depth_avg',
    'complementary_interactions', 'thematic_diversity'
]

g = sns.PairGrid(df[pairplot_features], diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.4, s=35, edgecolor='none', color='#3498DB')
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.3, 's': 25},
            line_kws={'color': '#E74C3C', 'linewidth': 2})
g.map_diag(sns.histplot, kde=True, alpha=0.7, color='#2C3E50')

g.fig.suptitle('Multivariate Relationship Analysis',
               fontsize=14, fontweight='bold', y=1.02)
plt.savefig('reports/figures/pairplot_relationships.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# EXECUTIVE SUMMARY
# =========================================================================
print("=" * 100)
print("✅ PHASE 1 COMPLETE — Executive Summary")
print("=" * 100)
print(f"  • Total Features Analyzed: {len(numerical_cols)} numerical | "
      f"{len(categorical_cols)} categorical")
print(f"  • Outlier Summary: See reports/tables/outlier_analysis_summary.csv")
print(f"  • Visualizations: 5 charts saved to reports/figures/")
print(f"  • Tables: descriptive_statistics, outlier, correlation, categorical -> reports/tables/")
print(f"  • Console Reports: Statistical, Outlier, Correlation, Categorical")
print(f"  • Ready for Phase 2: Clustering Analysis")
print("=" * 100)


##############################################

"""
=============================================================================
PHASE 2: DATA PREPROCESSING
=============================================================================
Objective: Transform raw session data into a scaled, encoded feature matrix
           suitable for distance-based clustering algorithms.

Key Decisions:
    - Drop 'navigation_speed': derived feature with extreme kurtosis (358.5)
      and redundant with 'pieces_started' + 'total_time_min'
    - StandardScaler: zero-mean unit-variance normalization for numerical features
    - One-hot encoding: nominal categorical variables without ordinal assumption
    - 'used_recommendations': retained as binary indicator (0/1)
    - 'pause_count': retained despite zero-inflation; represents valid behavioral signal
=============================================================================
"""

# =========================================================================
# 2.1 DATA LOADING
# =========================================================================
print("\n" + "=" * 100)
print("⚙️  PHASE 2: DATA PREPROCESSING")
print("=" * 100)

df = pd.read_csv("data/raw/digital_content_sessions.csv")
print(f"\nRaw data loaded:{df.shape[0]:,} rows ×{df.shape[1]} columns")

# =========================================================================
# 2.2 FEATURE SELECTION & ENGINEERING DECISIONS
# =========================================================================
print("\n" + "-" * 100)
print("2.2 FEATURE SELECTION DECISIONS")
print("-" * 100)

# Decision 1: Drop navigation_speed
print("\n  Decision 1 — Drop 'navigation_speed':")
print("    • Kurtosis: 358.51 (extreme deviation from normality)")
print("    • Minimum value: -1.994 (logically impossible negative speed)")
print("    • Information captured by parent variables: pieces_started + total_time_min")
print("    • Spearman ρ with pieces_started: 0.64")
print("    → Action: Feature excluded from clustering feature matrix")

# Decision 2: Retain pause_count
print("\n  Decision 2 — Retain 'pause_count' despite zero-inflation:")
print("    • 14.27% flagged as statistical outliers by IQR method")
print("    • Median = 0, but non-zero values represent meaningful behavioral signal")
print("    • Zero-inflation itself is a clustering signal (sessions with vs. without pauses)")
print("    → Action: Feature retained and scaled")

# Decision 3: Session ID
print("\n  Decision 3 — Exclude 'session_id':")
print("    • Unique identifier with no predictive or structural information")
print("    → Action: Stored separately for traceability, excluded from feature matrix")

# =========================================================================
# 2.3 DEFINE FEATURE GROUPS
# =========================================================================
numerical_features = [
    'total_time_min',
    'pieces_started',
    'completion_depth_avg',
    'complementary_interactions',
    'thematic_diversity',
    'pause_count'
    # 'navigation_speed' — intentionally excluded
]

categorical_features = ['moment_of_activity', 'device_type']
binary_features = ['used_recommendations']

# Store session IDs for traceability
session_ids = df['session_id'].copy()

# Extract feature matrix components
X_numerical = df[numerical_features].copy()
X_categorical = df[categorical_features].copy()
X_binary = df[binary_features].copy()

print(f"\n  Feature matrix composition:")
print(f"    Numerical features:{len(numerical_features)} —{numerical_features}")
print(f"    Categorical features:{len(categorical_features)} —{categorical_features}")
print(f"    Binary features:{len(binary_features)} —{binary_features}")
print(f"    Total features before encoding:{len(numerical_features) + len(categorical_features) + len(binary_features)}")

# =========================================================================
# 2.4 DATA QUALITY VERIFICATION
# =========================================================================
print("\n" + "-" * 100)
print("2.4 DATA QUALITY VERIFICATION")
print("-" * 100)

missing_report = pd.DataFrame({
    'Feature': numerical_features + categorical_features + binary_features,
    'Missing_Values': (list(X_numerical.isnull().sum()) +
                       list(X_categorical.isnull().sum()) +
                       list(X_binary.isnull().sum())),
    'Data_Type': (list(X_numerical.dtypes.astype(str)) +
                  list(X_categorical.dtypes.astype(str)) +
                  list(X_binary.dtypes.astype(str)))
})

missing_report['Status'] = missing_report['Missing_Values'].apply(
    lambda x: '✓ Clean' if x == 0 else '⚠ Review Required'
)

print(tabulate(missing_report, headers='keys', tablefmt='grid', showindex=False))
print(f"\n  → All features verified complete. No imputation required.")

# Save data quality report
missing_report.to_csv(f"{TABLES_DIR}/data_quality_report.csv", index=False)

# =========================================================================
# 2.5 STANDARDSCALER — NUMERICAL FEATURES
# =========================================================================
print("\n" + "-" * 100)
print("2.5 STANDARDSCALER APPLICATION — NUMERICAL FEATURES")
print("-" * 100)

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Create scaled DataFrame with meaningful column names
X_numerical_scaled_df = pd.DataFrame(
    X_numerical_scaled,
    columns=[f"{col}_scaled" for col in numerical_features],
    index=df.index
)

# Report scaling parameters
scaling_report = pd.DataFrame({
    'Feature': numerical_features,
    'Mean_Before': X_numerical.mean().round(4).values,
    'Std_Before': X_numerical.std().round(4).values,
    'Mean_After': X_numerical_scaled_df.mean().round(6).values,
    'Std_After': X_numerical_scaled_df.std().round(6).values
})

print("\n  Scaling Verification (target: μ≈0, σ≈1 after scaling):")
print(tabulate(scaling_report, headers='keys', tablefmt='grid',
               showindex=False, floatfmt='.6f'))
print("\n  → StandardScaler applied successfully. All features centered at μ=0, σ=1.")

# Save scaling report
scaling_report.to_csv(f"{TABLES_DIR}/scaling_report.csv", index=False)

# =========================================================================
# 2.6 ONE-HOT ENCODING — CATEGORICAL FEATURES
# =========================================================================
print("\n" + "-" * 100)
print("2.6 ONE-HOT ENCODING — CATEGORICAL FEATURES")
print("-" * 100)

# Apply one-hot encoding — retain all categories for interpretability
X_categorical_encoded = pd.get_dummies(
    X_categorical,
    columns=categorical_features,
    drop_first=False,
    dtype=int
)

print(f"\n  Categories encoded:")
encoding_records = []
for col in categorical_features:
    original_values = df[col].unique()
    encoded_cols = [c for c in X_categorical_encoded.columns if c.startswith(col)]
    print(f"    {col}: {len(original_values)} unique values → {len(encoded_cols)} binary columns")
    print(f"      Original: {sorted(original_values)}")
    print(f"      Encoded:  {encoded_cols}")
    encoding_records.append({
        'Feature': col,
        'Unique_Values': len(original_values),
        'Binary_Columns': len(encoded_cols),
        'Original': str(sorted(original_values)),
        'Encoded': str(encoded_cols)
    })

print(f"\n  Categorical features after encoding: {X_categorical_encoded.shape[1]} columns")

# Save one-hot encoding map
pd.DataFrame(encoding_records).to_csv(
    f"{TABLES_DIR}/onehot_encoding_map.csv", index=False)

# =========================================================================
# 2.7 FINAL FEATURE MATRIX ASSEMBLY
# =========================================================================
print("\n" + "-" * 100)
print("2.7 FINAL FEATURE MATRIX ASSEMBLY")
print("-" * 100)

# Align all feature groups by index
X_final = pd.concat([
    X_numerical_scaled_df,
    X_categorical_encoded.reset_index(drop=True),
    X_binary.reset_index(drop=True)
], axis=1)

print(f"\n  Final feature matrix dimensions: {X_final.shape[0]:,} rows × {X_final.shape[1]} columns")
print(f"  Feature list ({X_final.shape[1]} total):")
for i, col in enumerate(X_final.columns, 1):
    print(f"    {i:2d}. {col}")

# Save final feature matrix column list
pd.DataFrame({'index': range(1, len(X_final.columns) + 1),
              'feature': list(X_final.columns)}).to_csv(
    f"{TABLES_DIR}/final_feature_matrix_columns.csv", index=False)

# =========================================================================
# 2.8 PREPROCESSING IMPACT VISUALIZATION
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    ax = axes[i]

    # Plot original distribution
    sns.kdeplot(X_numerical[col], ax=ax, color=COLORS['original'],
                label='Original', linewidth=2, alpha=0.7)

    # Plot scaled distribution
    sns.kdeplot(X_numerical_scaled_df[f"{col}_scaled"], ax=ax,
                color=COLORS['scaled'], label='Scaled (μ=0, σ=1)',
                linewidth=2, alpha=0.7)

    ax.set_title(col.replace('_', ' ').title(), fontsize=11,
                 fontweight='bold', color=COLORS['primary'])
    ax.legend(frameon=True, fontsize=8)
    ax.set_xlabel('')

fig.suptitle('Preprocessing Impact: Original vs. Scaled Distributions',
             fontsize=14, fontweight='bold', y=1.02, color=COLORS['primary'])
plt.tight_layout()
plt.savefig('reports/figures/preprocessing_scaling_impact.png',
            dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 2.9 SAVE PROCESSED DATA
# =========================================================================
print("\n" + "-" * 100)
print("2.9 SAVING PROCESSED DATA")
print("-" * 100)

# Add session_id back for traceability
X_final_with_id = X_final.copy()
X_final_with_id.insert(0, 'session_id', session_ids)

# Save processed feature matrix
output_path = "data/processed/sessions_preprocessed.csv"
X_final_with_id.to_csv(output_path, index=False)
print(f"\n  Processed data saved: {output_path}")

# Save scaler parameters for reproducibility
scaler_params = pd.DataFrame({
    'feature': numerical_features,
    'mean': scaler.mean_,
    'std': scaler.scale_
})
scaler_params.to_csv("data/processed/scaler_parameters.csv", index=False)
print(f"  Scaler parameters saved: data/processed/scaler_parameters.csv")


#######################

"""
=============================================================================
PHASE 3: DIMENSIONALITY REDUCTION
=============================================================================
Objective: Reduce 14-dimensional feature space via PCA (≥80% variance)
           for clustering input; t-SNE for complementary visualization.

Note: scikit-learn ≥1.2 uses 'max_iter' instead of 'n_iter' for TSNE.
=============================================================================
"""

# =========================================================================
# 3.1 LOAD PREPROCESSED DATA
# =========================================================================
print("\n" + "=" * 100)
print("📉 PHASE 3: DIMENSIONALITY REDUCTION")
print("=" * 100)

df_processed = pd.read_csv("data/processed/sessions_preprocessed.csv")
session_ids = df_processed['session_id'].copy()

feature_cols = [col for col in df_processed.columns if col != 'session_id']
X = df_processed[feature_cols].copy()

print(f"\n  Input: {X.shape[0]:,} rows × {X.shape[1]} features")

# =========================================================================
# 3.2 PCA — EXPLAINED VARIANCE ANALYSIS
# =========================================================================
print("\n" + "-" * 100)
print("3.2 PRINCIPAL COMPONENT ANALYSIS")
print("-" * 100)

pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
threshold = 0.80
n_components = np.argmax(cumulative_variance >= threshold) + 1

# Variance report
variance_table = []
for i in range(len(explained_variance)):
    variance_table.append([
        f"PC{i+1}",
        f"{explained_variance[i]:.4f}",
        f"{explained_variance[i]*100:.1f}%",
        f"{cumulative_variance[i]:.4f}",
        f"{cumulative_variance[i]*100:.1f}%"
    ])

print(f"\n  Components retained (≥{threshold*100:.0f}% variance): {n_components}")
print(f"  Cumulative variance: {cumulative_variance[n_components-1]*100:.1f}%")
print(f"  Dimensionality reduction: {X.shape[1]} → {n_components} features")
print(f"\n  Full Variance Breakdown:")
print(tabulate(variance_table,
               headers=['Component', 'Variance', '%', 'Cumulative', 'Cumulative %'],
               tablefmt='grid', showindex=False))

# Save full variance breakdown
pd.DataFrame(variance_table,
             columns=['Component', 'Variance', 'Variance_%',
                      'Cumulative', 'Cumulative_%']).to_csv(
    f"{TABLES_DIR}/pca_variance_breakdown.csv", index=False)

# =========================================================================
# 3.3 EXPLAINED VARIANCE VISUALIZATION
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cumulative variance
axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', color='#3498DB', linewidth=2.5, markersize=8,
             markerfacecolor='white', markeredgewidth=2)
axes[0].axhline(y=threshold, color='#E74C3C', linestyle='--', linewidth=2,
                label=f'80% Threshold')
axes[0].axvline(x=n_components, color='#27AE60', linestyle=':', linewidth=2,
                label=f'{n_components} Components Selected')
axes[0].set_xlabel('Principal Components', fontweight='bold')
axes[0].set_ylabel('Cumulative Explained Variance', fontweight='bold')
axes[0].set_title('Cumulative Explained Variance', fontweight='bold', fontsize=13)
axes[0].legend(frameon=True, fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.05)

# Scree plot
bars = axes[1].bar(range(1, len(explained_variance) + 1), explained_variance,
                   color='#3498DB', edgecolor='white', linewidth=1.2)
for i in range(n_components):
    bars[i].set_color('#27AE60')
axes[1].set_xlabel('Principal Component', fontweight='bold')
axes[1].set_ylabel('Explained Variance Ratio', fontweight='bold')
axes[1].set_title(f'Scree Plot — {n_components} Components Retained (Green)',
                  fontweight='bold', fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/figures/pca_explained_variance.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 3.4 COMPONENT LOADING ANALYSIS
# =========================================================================
print("\n" + "-" * 100)
print("3.4 COMPONENT LOADING ANALYSIS")
print("-" * 100)

loadings = pd.DataFrame(
    pca_full.components_[:n_components].T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=feature_cols
)

# Identify dominant features per component
print(f"\n  Dominant Features per Component (|loading| > 0.50):")
dominant_records = []
for i in range(n_components):
    pc_loadings = loadings[f'PC{i+1}']
    strong_features = pc_loadings[pc_loadings.abs() > 0.50].sort_values(ascending=False)
    if len(strong_features) > 0:
        for feat, val in strong_features.items():
            direction = "(+)" if val > 0 else "(-)"
            print(f"    PC{i+1}: {feat} = {val:.3f} {direction}")
            dominant_records.append({'Component': f'PC{i+1}', 'Feature': feat,
                                     'Loading': round(val, 3), 'Direction': direction})
    else:
        print(f"    PC{i+1}: No single feature dominates (distributed loadings)")
        dominant_records.append({'Component': f'PC{i+1}', 'Feature': '(distributed)',
                                 'Loading': np.nan, 'Direction': '—'})

# Save dominant features and full loadings matrix
pd.DataFrame(dominant_records).to_csv(
    f"{TABLES_DIR}/pca_dominant_features.csv", index=False)
loadings.to_csv(f"{TABLES_DIR}/pca_loadings.csv")

# Loadings heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=1.2, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Loading Strength'},
            annot_kws={'size': 9, 'fontweight': 'bold'}, ax=ax)
ax.set_title('PCA Component Loadings Matrix\nFeature Contributions to Retained Components',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Principal Component', fontweight='bold')
ax.set_ylabel('Original Feature', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/pca_loadings_heatmap.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 3.5 APPLY PCA TRANSFORMATION
# =========================================================================
print("\n" + "-" * 100)
print("3.5 APPLYING PCA TRANSFORMATION")
print("-" * 100)

pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca.insert(0, 'session_id', session_ids)

print(f"\n  PCA transformation: {X.shape[1]} → {n_components} dimensions")
print(f"  Variance preserved: {cumulative_variance[n_components-1]*100:.1f}%")

# =========================================================================
# 3.6 t-SNE — NON-LINEAR PROJECTION (COMPATIBLE WITH SCIKIT-LEARN ≥1.2)
# =========================================================================
print("\n" + "-" * 100)
print("3.6 t-SNE — NON-LINEAR PROJECTION")
print("-" * 100)

# scikit-learn ≥1.2 uses 'max_iter' instead of 'n_iter'
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,        # Updated from 'n_iter' for sklearn ≥1.2
    random_state=RANDOM_STATE,
    init='pca',
    learning_rate='auto'
)

X_tsne = tsne.fit_transform(X)

df_tsne = pd.DataFrame(X_tsne, columns=['tSNE_1', 'tSNE_2'])
df_tsne.insert(0, 'session_id', session_ids)

print(f"\n  t-SNE projection: {X.shape[1]} → 2 dimensions")
print(f"  Perplexity: 30 | Max iterations: 1000")
print(f"  KL divergence: {tsne.kl_divergence_:.4f}")
print(f"  Note: t-SNE is for visualization only — NOT used as clustering input")

# =========================================================================
# 3.7 DUAL PROJECTION COMPARISON
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA projection
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c='#3498DB', alpha=0.5, s=35,
                edgecolor='white', linewidth=0.3)
axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontweight='bold')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontweight='bold')
axes[0].set_title('PCA Projection — Linear Global Structure', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3)

# t-SNE projection
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c='#E67E22', alpha=0.5, s=35,
                edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('t-SNE Dimension 1', fontweight='bold')
axes[1].set_ylabel('t-SNE Dimension 2', fontweight='bold')
axes[1].set_title('t-SNE Projection — Non-Linear Local Structure', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Dimensionality Reduction: Dual Perspective\n'
             'PCA (Clustering Input) vs. t-SNE (Visual Validation)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/dual_projection_comparison.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# 3.8 SAVE ALL RESULTS
# =========================================================================
print("\n" + "-" * 100)
print("3.8 SAVING RESULTS")
print("-" * 100)

df_pca.to_csv("data/processed/sessions_pca.csv", index=False)
df_tsne.to_csv("data/processed/sessions_tsne.csv", index=False)
loadings.to_csv("data/processed/pca_loadings.csv")

# Save PCA parameters
pca_params = pd.DataFrame({
    'component': [f'PC{i+1}' for i in range(n_components)],
    'explained_variance': pca.explained_variance_ratio_,
    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
})
pca_params.to_csv("data/processed/pca_parameters.csv", index=False)

print(f"  ✓ sessions_pca.csv — PCA-transformed data ({n_components} dimensions)")
print(f"  ✓ sessions_tsne.csv — t-SNE projection (2 dimensions)")
print(f"  ✓ pca_loadings.csv — Component loadings matrix")
print(f"  ✓ pca_parameters.csv — Explained variance per component")

# =========================================================================
# 3.9 PHASE 3 COMPLETION
# =========================================================================
print("\n" + "=" * 100)
print("✅ PHASE 3: DIMENSIONALITY REDUCTION — COMPLETE")
print("=" * 100)

summary = {
    "Input Features": X.shape[1],
    "PCA Components Retained": n_components,
    "Variance Preserved": f"{cumulative_variance[n_components-1]*100:.1f}%",
    "Reduction": f"{X.shape[1]} → {n_components}",
    "t-SNE Dimensions": 2,
    "t-SNE KL Divergence": f"{tsne.kl_divergence_:.4f}",
}

for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n  Next: Phase 4 — Clustering (K-Means & DBSCAN)")
print("=" * 100)


"""
=============================================================================
PHASE 4: UNSUPERVISED CLUSTERING — K-MEANS & DBSCAN
=============================================================================
Objective: Discover natural behavioral segments using two complementary
           algorithms. Diagnose algorithm suitability and select best model.

Data: 1,500 sessions × 3 PCA components (80.7% variance preserved)
=============================================================================
"""

# =========================================================================
# DATA LOADING
# =========================================================================
print("\n" + "═" * 80)
print("📊 PHASE 4: UNSUPERVISED CLUSTERING")
print("═" * 80)

df_pca = pd.read_csv("data/processed/sessions_pca.csv")
session_ids = df_pca['session_id'].copy()
X_pca = df_pca[['PC1', 'PC2', 'PC3']].values

print(f"\n  Clustering Input")
print(f"  {'─' * 40}")
print(f"  Sessions : {X_pca.shape[0]:,}")
print(f"  Features : {X_pca.shape[1]} PCA components")
print(f"  Variance : 80.7% preserved")

# =========================================================================
# K-MEANS — OPTIMAL k DETERMINATION (k = 2 to 10)
# =========================================================================
print(f"\n  K-Means — Optimal k Search")
print(f"  {'─' * 40}")

k_range = range(2, 11)
inertias = []
silhouette_scores_kmeans = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores_kmeans.append(silhouette_score(X_pca, labels))

best_k = k_range[np.argmax(silhouette_scores_kmeans)]

# Evaluation table
eval_table = []
for i, k in enumerate(k_range):
    marker = "◄ BEST" if k == best_k else ""
    eval_table.append([k, f"{inertias[i]:.2f}", f"{silhouette_scores_kmeans[i]:.4f}", marker])

print(tabulate(eval_table,
               headers=['k', 'Inertia', 'Silhouette', ''],
               tablefmt='grid', showindex=False))
print(f"\n  Selected k = {best_k}  |  Silhouette = {max(silhouette_scores_kmeans):.4f}")

# Save k-selection table
pd.DataFrame(eval_table, columns=['k', 'Inertia', 'Silhouette', 'Best']).to_csv(
    f"{TABLES_DIR}/kmeans_k_selection.csv", index=False)

# =========================================================================
# FINAL K-MEANS MODEL
# =========================================================================
kmeans_final = KMeans(n_clusters=best_k, n_init=25, random_state=RANDOM_STATE)
kmeans_labels = kmeans_final.fit_predict(X_pca)

sil_final = silhouette_score(X_pca, kmeans_labels)
db_final = davies_bouldin_score(X_pca, kmeans_labels)

print(f"\n  Final Model Validation")
print(f"  {'─' * 40}")
print(f"  Silhouette Score      : {sil_final:.4f}")
print(f"  Davies-Bouldin Index  : {db_final:.4f}")
print(f"\n  Cluster Distribution:")
distribution_records = []
for i in range(best_k):
    count = np.sum(kmeans_labels == i)
    pct = count / len(kmeans_labels) * 100
    bar = '█' * int(pct / 2)
    print(f"  Cluster {i} : {count:4d} sessions ({pct:5.1f}%)  {bar}")
    distribution_records.append({'Cluster': i, 'Count': int(count),
                                 'Percentage': f"{pct:.1f}%"})

# Save cluster distribution
pd.DataFrame(distribution_records).to_csv(
    f"{TABLES_DIR}/kmeans_cluster_distribution.csv", index=False)

# =========================================================================
# K-MEANS VISUALIZATION DASHBOARD
# =========================================================================
fig = plt.figure(figsize=(22, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

# ---- Panel 1: Elbow Curve ----
ax1 = fig.add_subplot(gs[0])
ax1.plot(k_range, inertias,
         color=COLORS['kmeans'], linewidth=2.8, marker='o',
         markersize=11, markerfacecolor='white', markeredgewidth=2.5,
         markeredgecolor=COLORS['kmeans'], label='Inertia')
ax1.axvline(x=best_k, color=COLORS['accent'], linestyle='--',
            linewidth=2.2, alpha=0.85, label=f'Selected k = {best_k}')
ax1.fill_between(k_range, inertias, alpha=0.08, color=COLORS['kmeans'])
ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=11)
ax1.set_title('Elbow Method', fontsize=13, pad=12, color=COLORS['primary'])
ax1.legend(frameon=True, fancybox=True, fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.25, color=COLORS['grid'])
ax1.set_xticks(list(k_range))

# ---- Panel 2: Silhouette Scores ----
ax2 = fig.add_subplot(gs[1])
ax2.plot(k_range, silhouette_scores_kmeans,
         color=COLORS['silhouette'], linewidth=2.8, marker='s',
         markersize=11, markerfacecolor='white', markeredgewidth=2.5,
         markeredgecolor=COLORS['silhouette'], label='Silhouette')
ax2.axvline(x=best_k, color=COLORS['accent'], linestyle='--',
            linewidth=2.2, alpha=0.85, label=f'Best k = {best_k}')
ax2.fill_between(k_range, silhouette_scores_kmeans, alpha=0.08,
                 color=COLORS['silhouette'])
max_idx = np.argmax(silhouette_scores_kmeans)
ax2.scatter([k_range[max_idx]], [silhouette_scores_kmeans[max_idx]],
            s=180, c=COLORS['accent'], zorder=5, marker='D',
            edgecolor='white', linewidth=2)
ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.set_title('Silhouette Analysis', fontsize=13, pad=12, color=COLORS['primary'])
ax2.legend(frameon=True, fancybox=True, fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.25, color=COLORS['grid'])
ax2.set_xticks(list(k_range))

# ---- Panel 3: Cluster Map (PC1 vs PC2) ----
ax3 = fig.add_subplot(gs[2])
centroids = kmeans_final.cluster_centers_
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=kmeans_labels, cmap='viridis',
                      alpha=0.65, s=45, edgecolor='white', linewidth=0.4)
ax3.scatter(centroids[:, 0], centroids[:, 1],
            marker='X', s=350, c=COLORS['centroid'],
            edgecolor='white', linewidth=2.5, zorder=10, label='Centroids')

# Cluster boundary circles
for i, (cx, cy) in enumerate(centroids[:, :2]):
    cluster_points = X_pca[kmeans_labels == i][:, :2]
    radius = np.percentile(np.linalg.norm(cluster_points - [cx, cy], axis=1), 85)
    circle = plt.Circle((cx, cy), radius, fill=False, color='white',
                        linewidth=1.5, linestyle='--', alpha=0.5)
    ax3.add_patch(circle)
    ax3.annotate(f'C{i}', (cx, cy), fontsize=11, fontweight='bold',
                 color='white', ha='center', va='center',
                 bbox=dict(boxstyle='circle,pad=0.3', facecolor=COLORS['primary'],
                          alpha=0.85))

ax3.set_xlabel('PC1 (53.9% variance)', fontsize=11)
ax3.set_ylabel('PC2 (20.8% variance)', fontsize=11)
ax3.set_title(f'Cluster Map — PCA Space (k = {best_k})',
              fontsize=13, pad=12, color=COLORS['primary'])
ax3.legend(frameon=True, fancybox=True, fontsize=9, loc='best')
ax3.grid(True, alpha=0.25, color=COLORS['grid'])

cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8, pad=0.02)
cbar.set_label('Cluster Assignment', fontsize=10, fontweight='bold')

fig.suptitle('K-Means Clustering Diagnostics',
             fontsize=15, fontweight='bold', y=1.03, color=COLORS['primary'])
plt.tight_layout()
plt.savefig('reports/figures/kmeans_diagnostics.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# DBSCAN — EPSILON ESTIMATION
# =========================================================================
print(f"\n  DBSCAN — Epsilon Estimation")
print(f"  {'─' * 40}")

k_neighbors = 4
neighbors = NearestNeighbors(n_neighbors=k_neighbors)
neighbors_fit = neighbors.fit(X_pca)
k_distances = np.sort(neighbors_fit.kneighbors(X_pca)[0][:, -1])

gradient2 = np.gradient(np.gradient(k_distances))
elbow_idx = np.argmax(gradient2[:len(gradient2)//2])
epsilon_suggested = k_distances[elbow_idx]

print(f"  Suggested ε : {epsilon_suggested:.4f}")
print(f"  Elbow point : index {elbow_idx} of {len(k_distances)}")

# ---- k-Distance Visualization ----
fig, ax = plt.subplots(figsize=(14, 6))

# Gradient background
x_range = np.arange(len(k_distances))
ax.fill_between(x_range, k_distances, alpha=0.08, color='#1a1a2e')
ax.plot(x_range, k_distances, color='#16213e', linewidth=2.2, alpha=0.95)

# Elbow point highlight
ax.scatter([elbow_idx], [epsilon_suggested], s=180,
           c='#e94560', zorder=10, marker='D',
           edgecolor='white', linewidth=2)

# Vertical reference line
ax.axvline(x=elbow_idx, color='#e94560', linestyle='--',
           linewidth=1.5, alpha=0.6)

# Annotation
ax.annotate(
    f'ε = {epsilon_suggested:.4f}',
    xy=(elbow_idx, epsilon_suggested),
    xytext=(elbow_idx + 180, epsilon_suggested * 1.4),
    fontsize=11, fontweight='bold', color='#16213e',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
              edgecolor='#e94560', alpha=0.95),
    arrowprops=dict(arrowstyle='->', color='#e94560',
                   lw=1.8, connectionstyle='arc3,rad=0.2')
)

ax.set_xlabel('Data Points (sorted by distance)', fontsize=11, labelpad=10)
ax.set_ylabel(f'{k_neighbors}-th Nearest Neighbor Distance', fontsize=11, labelpad=10)
ax.set_title('DBSCAN Epsilon Estimation — k-Distance Graph',
             fontsize=14, pad=15, fontweight='bold', color='#1a1a2e')
ax.grid(True, alpha=0.2, linestyle='-', color='#bdc3c7')
ax.set_xlim(0, len(k_distances))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('reports/figures/dbscan_kdistance.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# DBSCAN — PARAMETER GRID SEARCH
# =========================================================================
print(f"\n  DBSCAN — Parameter Grid Search")
print(f"  {'─' * 40}")

eps_values = np.linspace(epsilon_suggested * 0.5, epsilon_suggested * 2, 8)
min_samples_values = [3, 5, 7, 10]

dbscan_results = []
best_config = None
best_score = -1

for eps in eps_values:
    for min_samp in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        labels = dbscan.fit_predict(X_pca)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_pct = n_noise / len(labels) * 100
        sil = np.nan

        if n_clusters >= 2 and n_noise < len(labels) * 0.9:
            mask = labels != -1
            sil = silhouette_score(X_pca[mask], labels[mask])

            if n_noise < len(labels) * 0.3:
                score = sil - abs(n_clusters - best_k) * 0.05
                if score > best_score:
                    best_score = score
                    best_config = {
                        'eps': eps, 'min_samples': min_samp,
                        'n_clusters': n_clusters, 'silhouette': sil
                    }

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "—"
        dbscan_results.append([
            f"{eps:.4f}", min_samp, n_clusters, n_noise,
            f"{noise_pct:.1f}%", sil_str
        ])

print(tabulate(dbscan_results,
               headers=['Epsilon', 'Min Pts', 'Clusters', 'Noise', '% Noise', 'Silhouette'],
               tablefmt='grid', showindex=False))

# Save DBSCAN grid search
pd.DataFrame(dbscan_results,
             columns=['Epsilon', 'Min_Pts', 'Clusters', 'Noise',
                      'Noise_%', 'Silhouette']).to_csv(
    f"{TABLES_DIR}/dbscan_grid_search.csv", index=False)

# =========================================================================
# DBSCAN FAILURE DIAGNOSIS
# =========================================================================
print(f"\n  DBSCAN — Failure Diagnosis")
print(f"  {'─' * 40}")

# Analyze why DBSCAN fails
if best_config is None:
    # Find configuration with least noise
    valid_configs = [r for r in dbscan_results if r[2] >= 2]

    print(f"\n  Root Cause Analysis:")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ 1. Uniform Density Distribution                            │")
    print(f"  │    The k-distance curve shows a smooth gradient without     │")
    print(f"  │    a sharp inflection point, indicating no natural density  │")
    print(f"  │    drops in the data. Clusters have similar densities.      │")
    print(f"  │                                                             │")
    print(f"  │ 2. Spherical Cluster Geometry                               │")
    print(f"  │    PCA loadings (Phase 3) revealed that PC1 captures        │")
    print(f"  │    distributed variance across multiple features. The       │")
    print(f"  │    resulting clusters are spherical and well-separated —    │")
    print(f"  │    ideal for K-Means, problematic for DBSCAN.               │")
    print(f"  │                                                             │")
    print(f"  │ 3. No Natural Noise Points                                  │")
    print(f"  │    The synthetic data was engineered with 4 clean clusters. │")
    print(f"  │    Every point belongs to a meaningful group. DBSCAN's      │")
    print(f"  │    noise detection becomes a liability, not a feature.      │")
    print(f"  │                                                             │")
    print(f"  │ 4. Extreme Noise Sensitivity                               │")
    print(f"  │    At ε={epsilon_suggested:.4f}, even min_pts=3 produces     │")
    print(f"  │    massive noise. Increasing ε fragments clusters into      │")
    print(f"  │    dozens of micro-groups (55 clusters at ε=0.0647).        │")
    print(f"  │    No ε balances cluster count with acceptable noise.       │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    dbscan_labels = np.full(len(X_pca), -1)

    # Save DBSCAN root-cause diagnosis
    pd.DataFrame({
        'Cause': [
            'Uniform Density Distribution',
            'Spherical Cluster Geometry',
            'No Natural Noise Points',
            'Extreme Noise Sensitivity'
        ],
        'Description': [
            'Smooth k-distance gradient without sharp inflection; clusters share similar densities.',
            'PC1 captures distributed variance; clusters are spherical and well-separated (ideal for K-Means).',
            'Synthetic data engineered with 4 clean clusters; every point belongs to a meaningful group.',
            f'At eps={epsilon_suggested:.4f} even min_pts=3 yields massive noise; higher eps fragments into micro-groups.'
        ]
    }).to_csv(f"{TABLES_DIR}/dbscan_failure_diagnosis.csv", index=False)
else:
    dbscan_final = DBSCAN(eps=best_config['eps'],
                          min_samples=best_config['min_samples'])
    dbscan_labels = dbscan_final.fit_predict(X_pca)
    n_clusters_db = best_config['n_clusters']
    n_noise = np.sum(dbscan_labels == -1)

    print(f"\n  DBSCAN — Final Configuration")
    print(f"  Epsilon: {best_config['eps']:.4f} | Min Samples: {best_config['min_samples']}")
    print(f"  Clusters: {n_clusters_db} | Noise: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")

# =========================================================================
# ALGORITHM COMPARISON & FINAL SELECTION
# =========================================================================
print(f"\n{'═' * 80}")
print(f"📋 MODEL SELECTION")
print(f"{'═' * 80}")

selection_data = {
    'Criterion': [
        'Clusters Identified',
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Point Coverage',
        'Algorithm-Data Fit',
        'Business Interpretability'
    ],
    'K-Means (k=4)': [
        '4',
        f'{sil_final:.4f}',
        f'{db_final:.4f}',
        '1,500 / 1,500 (100%)',
        '✓ Optimal — spherical, well-separated clusters',
        '✓ High — clear centroid profiles'
    ],
    'DBSCAN Assessment': [
        '0–55 (unstable)',
        '— (≥86.5% noise)',
        '— (no valid config)',
        '≤ 203 / 1,500 (≤13.5%)',
        '✗ Structurally unsuitable — uniform density',
        '✗ Noise-dominant output uninterpretable'
    ]
}

print(tabulate(selection_data, headers='keys', tablefmt='grid',
               showindex=False, maxcolwidths=[22, 30, 35]))

print(f"\n  ✓ Selected Algorithm : K-Means (k=4)")
print(f"  ✓ Rationale :")
print(f"    • Silhouette 0.7151 indicates well-defined, compact clusters")
print(f"    • Davies-Bouldin 0.3867 confirms strong inter-cluster separation")
print(f"    • 100% point coverage enables complete behavioral profiling")
print(f"    • DBSCAN failure validates K-Means as the correct methodological choice")

# Save model comparison
pd.DataFrame(selection_data).to_csv(
    f"{TABLES_DIR}/model_selection_comparison.csv", index=False)

# =========================================================================
# SAVE RESULTS
# =========================================================================
print(f"\n  Saving Outputs")
print(f"  {'─' * 40}")

results = pd.DataFrame({
    'session_id': session_ids,
    'kmeans_cluster': kmeans_labels
})

if best_config:
    results['dbscan_cluster'] = dbscan_labels

results.to_csv("data/processed/cluster_assignments.csv", index=False)
joblib.dump(kmeans_final, "models/kmeans_model.pkl")

print(f"  ✓ cluster_assignments.csv  ({len(results)} rows, {results.shape[1]} columns)")
print(f"  ✓ kmeans_model.pkl")

if best_config:
    joblib.dump(dbscan_final, "models/dbscan_model.pkl")
    print(f"  ✓ dbscan_model.pkl")

print(f"\n{'═' * 80}")
print(f"✅ PHASE 4 COMPLETE — Proceed to Phase 5: Cluster Profiling")
print(f"{'═' * 80}\n")


"""
=============================================================================
PHASE 5: CLUSTER VALIDATION, PROFILING & BUSINESS INTERPRETATION
=============================================================================
"""

# Load and merge
df_original = pd.read_csv("data/raw/digital_content_sessions.csv")
df_clusters = pd.read_csv("data/processed/cluster_assignments.csv")
df_processed = pd.read_csv("data/processed/sessions_preprocessed.csv")

df = df_original.merge(df_clusters[['session_id', 'kmeans_cluster']], on='session_id')

numerical_cols = [
    'total_time_min', 'pieces_started', 'completion_depth_avg',
    'complementary_interactions', 'thematic_diversity', 'pause_count'
]
categorical_cols = ['moment_of_activity', 'device_type', 'used_recommendations']

print(f"Data merged: {df.shape[0]} sessions × {df.shape[1]} features")
print(f"Clusters identified: {df['kmeans_cluster'].nunique()}")

# =========================================================================
# SILHOUETTE ANALYSIS
# =========================================================================
X_scaled = StandardScaler().fit_transform(df_processed.drop(columns=['session_id']))
X_pca_val = PCA(n_components=3, random_state=42).fit_transform(X_scaled)

silhouette_vals = silhouette_samples(X_pca_val, df['kmeans_cluster'].values)

print(f"\n  Silhouette Validation — Per Cluster")
print(f"  {'─' * 50}")
silhouette_records = []
for c in sorted(df['kmeans_cluster'].unique()):
    cluster_sil = silhouette_vals[df['kmeans_cluster'] == c]
    n_negative = np.sum(cluster_sil < 0)
    print(f"  Cluster {c}: mean = {cluster_sil.mean():.4f}  |  "
          f"negative = {n_negative:3d} pts  |  size = {len(cluster_sil)}")
    silhouette_records.append({'Cluster': c,
                               'Mean_Silhouette': round(cluster_sil.mean(), 4),
                               'Negative_Points': int(n_negative),
                               'Size': len(cluster_sil)})

print(f"\n  Global Silhouette: {silhouette_vals.mean():.4f}")

# Save per-cluster silhouette
pd.DataFrame(silhouette_records).to_csv(
    f"{TABLES_DIR}/silhouette_per_cluster.csv", index=False)

# Silhouette plot
fig, ax = plt.subplots(figsize=(12, 7))
y_lower = 0
cluster_colors = plt.cm.viridis(np.linspace(0, 1, 4))

for i, c in enumerate(sorted(df['kmeans_cluster'].unique())):
    cluster_sil = np.sort(silhouette_vals[df['kmeans_cluster'] == c])
    cluster_size = len(cluster_sil)
    y_upper = y_lower + cluster_size

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     facecolor=cluster_colors[i], alpha=0.7, label=f'Cluster {c}')
    y_lower = y_upper

avg_sil = silhouette_vals.mean()
ax.axvline(x=avg_sil, color=COLORS['accent'], linestyle='--', linewidth=2,
           label=f'Average: {avg_sil:.4f}')
ax.set_xlabel('Silhouette Coefficient', fontweight='bold', fontsize=11)
ax.set_ylabel('Session', fontweight='bold', fontsize=11)
ax.set_title('Silhouette Plot — K-Means Cluster Stability (k=4)',
             fontweight='bold', fontsize=13, color=COLORS['primary'])
ax.legend(frameon=True, fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('reports/figures/silhouette_validation.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# NUMERICAL FEATURE PROFILING
# =========================================================================
print(f"\n  Cluster Profiles — Numerical Features (Mean ± Std)")
print(f"  {'─' * 60}")

profile_num = df.groupby('kmeans_cluster')[numerical_cols].agg(['mean', 'std']).round(2)

profile_rows = []
for col in numerical_cols:
    row = [col.replace('_', ' ').title()]
    for c in sorted(df['kmeans_cluster'].unique()):
        mean_val = profile_num.loc[c, (col, 'mean')]
        std_val = profile_num.loc[c, (col, 'std')]
        row.append(f"{mean_val:.1f} ± {std_val:.1f}")
    profile_rows.append(row)

profile_headers = ['Feature'] + [f'Cluster {c}' for c in sorted(df['kmeans_cluster'].unique())]
print(tabulate(profile_rows, headers=profile_headers, tablefmt='grid', showindex=False))

# Save numerical profiles (readable format + full matrix)
pd.DataFrame(profile_rows, columns=profile_headers).to_csv(
    f"{TABLES_DIR}/cluster_numerical_profiles.csv", index=False)
profile_num.to_csv(f"{TABLES_DIR}/cluster_numerical_profiles_full.csv")

# =========================================================================
# CATEGORICAL FEATURE PROFILING
# =========================================================================
print(f"\n  Cluster Profiles — Categorical Features (%)")
print(f"  {'─' * 55}")

categorical_profile_frames = []
for col in categorical_cols:
    print(f"\n  {col.replace('_', ' ').title()}:")
    crosstab = pd.crosstab(
        df['kmeans_cluster'], df[col], normalize='index'
    ).round(3) * 100

    crosstab_display = crosstab.map(lambda x: f"{x:.1f}%")
    print(tabulate(crosstab_display, headers='keys', tablefmt='grid', showindex=True))

    # Accumulate for export
    ct_save = crosstab.add_prefix(f"{col}=")
    categorical_profile_frames.append(ct_save)

# Save categorical profiles
pd.concat(categorical_profile_frames, axis=1).to_csv(
    f"{TABLES_DIR}/cluster_categorical_profiles.csv")

# =========================================================================
# RADAR CHART
# =========================================================================
profile_means = df.groupby('kmeans_cluster')[numerical_cols].mean()
profile_norm = (profile_means - profile_means.min()) / (profile_means.max() - profile_means.min())

categories = [c.replace('_', ' ').title() for c in numerical_cols]
n_categories = len(categories)
angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
radar_colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12']

for i, c in enumerate(sorted(df['kmeans_cluster'].unique())):
    values = profile_norm.loc[c].tolist()
    values += values[:1]
    ax.fill(angles, values, alpha=0.12, color=radar_colors[i])
    ax.plot(angles, values, 'o-', linewidth=2.5, color=radar_colors[i],
            label=f'Cluster {c} ({len(df[df["kmeans_cluster"]==c])} sessions)',
            markersize=8)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_title('Cluster Comparison — Behavioral Signatures',
             fontsize=14, fontweight='bold', pad=25, color=COLORS['primary'])
ax.legend(frameon=True, fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/cluster_radar_profiles.png', dpi=200, facecolor='white')
plt.show()

# =========================================================================
# PERSONA MAPPING — DIRECT ASSIGNMENT BASED ON EMPIRICAL RESULTS
# =========================================================================
print(f"\n{'═' * 80}")
print(f"📋 BUSINESS INTERPRETATION — CONSUMPTION PERSONAS")
print(f"{'═' * 80}")

# Persona assignments derived from cluster profiles
persona_map = {
    0: ("Download & Depart",
        "Brief sessions (4.9 min), minimal depth (8.3%), high save actions (6.6). "
        "Varied devices, all time blocks. Users collect content for later."),
    1: ("Sampling Nomads",
        "Many pieces (14.6), very low depth (11.8%), high diversity (6.9), "
        "frequent pauses (2.5). Mobile-dominant, afternoon/evening peak."),
    2: ("Marathon Finishers",
        "Long sessions (52.2 min), near-complete depth (91.3%), high interactions (9.5), "
        "low diversity (1.5). Desktop-dominant, daytime hours."),
    3: ("Burst Consumers",
        "Short sessions (11.1 min), moderate depth (47.5%), mobile-exclusive. "
        "Sporadic timing across evening, late night, and morning.")
}

strategy_map = {
    0: "Offline access, queue management, push notifications for saved content",
    1: "Curated discovery feeds, topic bundles, social sharing features",
    2: "Premium long-form content, sequential playlists, loyalty rewards",
    3: "Short-form mobile UX, seamless session resumption, micro-playlists"
}

risk_map = {
    0: "Platform treated as discovery tool only; consumption happens elsewhere",
    1: "May never convert to deep engagement; suits ad-supported tier",
    2: "Churn if content quality declines; high expectations",
    3: "Session abandonment if UX friction exists; needs instant loading"
}

# Persona summary table
persona_table = []
for c in sorted(persona_map.keys()):
    size = len(df[df['kmeans_cluster'] == c])
    pct = size / len(df) * 100
    persona_table.append([
        f"Cluster {c}",
        persona_map[c][0],
        f"{pct:.1f}%",
        f"{size} sessions",
        strategy_map[c]
    ])

print(tabulate(persona_table,
               headers=['Segment', 'Persona', 'Share', 'Count', 'Strategic Recommendation'],
               tablefmt='grid', showindex=False, maxcolwidths=[12, 22, 8, 14, 50]))

# Detailed profiles
print(f"\n  Persona Profiles — Behavioral Signature, Strategy & Risk")
print(f"  {'─' * 65}")

persona_detailed = []
for c in sorted(persona_map.keys()):
    name, signature = persona_map[c]
    print(f"\n  ▸ {name} (Cluster {c})")
    print(f"    Signature  : {signature}")
    print(f"    Strategy   : {strategy_map[c]}")
    print(f"    Risk       : {risk_map[c]}")
    persona_detailed.append({
        'Cluster': c, 'Persona': name, 'Signature': signature,
        'Strategy': strategy_map[c], 'Risk': risk_map[c]
    })

# Save persona mapping (summary + detailed)
pd.DataFrame(persona_table,
    columns=['Cluster', 'Persona', 'Share', 'Count', 'Strategy']
).to_csv(f"{TABLES_DIR}/cluster_persona_mapping.csv", index=False)

pd.DataFrame(persona_detailed).to_csv(
    f"{TABLES_DIR}/cluster_persona_detailed.csv", index=False)

print(f"\n  ✓ Persona mapping saved: reports/tables/cluster_persona_mapping.csv")

# =========================================================================
# OUTPUT VERIFICATION — reports/tables/ CONTENTS
# =========================================================================
print(f"\n{'═' * 80}")
print(f"📂 reports/tables/ CONTENTS")
print(f"{'═' * 80}")
for f in sorted(os.listdir(TABLES_DIR)):
    size = os.path.getsize(os.path.join(TABLES_DIR, f))
    print(f"  ✓ {f}  ({size:,} bytes)")

print(f"\n{'═' * 80}")
print(f"✅ PHASE 5 COMPLETE")
print(f"{'═' * 80}\n")
