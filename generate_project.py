"""
=============================================================================
DIGITAL CONTENT CONSUMPTION PATTERNS - SYNTHETIC DATA GENERATOR
=============================================================================
Project: Customer Analytics Portfolio - Unsupervised Learning
Objective: Generate synthetic session-level data from a digital content
           platform to identify behavioral consumption patterns via
           clustering algorithms (K-Means, DBSCAN).

Underlying Behavioral Clusters:
    1. Marathon Finishers: Complete full pieces with high depth,
       long sessions, strong complementary interactions.
    2. Sampling Nomads: Browse many pieces superficially across
       diverse categories, high navigation speed.
    3. Burst Consumers: Short, intense sessions, moderate completion
       depth, sporadic timing, mobile-heavy.
    4. Download & Depart: Minimal in-session consumption, high
       complementary interactions (downloads/saves), very low
       completion depth, brief sessions.

Variables reflect user interaction without demographic or location
identifiers. Data designed to contain natural clusters representing
distinct "digital personas".

Recommended rows: 1500 (sufficient density for DBSCAN to identify
natural clusters while remaining computationally efficient)
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# =============================================================================
# CREATE PROJECT DIRECTORY STRUCTURE
# =============================================================================
def create_project_structure():
    """Create the project's folder structure for deliverables."""
    folders = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "reports/figures",
        "reports/tables",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  Created: {folder}/")
    return True


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
def generate_synthetic_sessions(n_sessions=1500, seed=42):
    """Generate synthetic session data with 4 latent behavioral clusters."""
    np.random.seed(seed)

    proportions = [0.25, 0.30, 0.25, 0.20]
    cluster_sizes = [int(n_sessions * p) for p in proportions]
    cluster_sizes[-1] = n_sessions - sum(cluster_sizes[:-1])

    all_data = []

    # ---- CLUSTER 0: MARATHON FINISHERS ----
    n_c0 = cluster_sizes[0]
    data_c0 = {
        "total_time_min": np.random.normal(52, 9, n_c0),
        "pieces_started": np.random.randint(2, 5, n_c0),
        "completion_depth_avg": np.random.normal(91, 5, n_c0),
        "complementary_interactions": np.random.randint(6, 14, n_c0),
        "thematic_diversity": np.random.randint(1, 3, n_c0),
        "pause_count": np.random.randint(0, 1, n_c0),
        "moment_of_activity": np.random.choice(
            ["morning", "afternoon"], n_c0, p=[0.50, 0.50]),
        "device_type": np.random.choice(
            ["desktop", "tablet"], n_c0, p=[0.90, 0.10]),
        "used_recommendations": np.random.choice([0, 1], n_c0, p=[0.10, 0.90]),
    }
    data_c0["navigation_speed"] = data_c0["pieces_started"] / data_c0["total_time_min"]
    data_c0["completion_depth_avg"] = np.clip(data_c0["completion_depth_avg"], 80, 100)
    data_c0["total_time_min"] = np.clip(data_c0["total_time_min"], 35, 80)
    all_data.append(pd.DataFrame(data_c0))

    # ---- CLUSTER 1: SAMPLING NOMADS ----
    n_c1 = cluster_sizes[1]
    data_c1 = {
        "total_time_min": np.random.normal(22, 5, n_c1),
        "pieces_started": np.random.randint(10, 20, n_c1),
        "completion_depth_avg": np.random.normal(12, 5, n_c1),
        "complementary_interactions": np.random.randint(0, 2, n_c1),
        "thematic_diversity": np.random.randint(5, 10, n_c1),
        "pause_count": np.random.randint(1, 5, n_c1),
        "moment_of_activity": np.random.choice(
            ["afternoon", "evening"], n_c1, p=[0.40, 0.60]),
        "device_type": np.random.choice(
            ["mobile", "tablet"], n_c1, p=[0.85, 0.15]),
        "used_recommendations": np.random.choice([0, 1], n_c1, p=[0.75, 0.25]),
    }
    data_c1["navigation_speed"] = data_c1["pieces_started"] / data_c1["total_time_min"]
    data_c1["completion_depth_avg"] = np.clip(data_c1["completion_depth_avg"], 4, 25)
    data_c1["total_time_min"] = np.clip(data_c1["total_time_min"], 10, 36)
    all_data.append(pd.DataFrame(data_c1))

    # ---- CLUSTER 2: BURST CONSUMERS ----
    n_c2 = cluster_sizes[2]
    data_c2 = {
        "total_time_min": np.random.normal(11, 3, n_c2),
        "pieces_started": np.random.randint(3, 7, n_c2),
        "completion_depth_avg": np.random.normal(48, 12, n_c2),
        "complementary_interactions": np.random.randint(1, 4, n_c2),
        "thematic_diversity": np.random.randint(1, 4, n_c2),
        "pause_count": np.random.randint(0, 2, n_c2),
        "moment_of_activity": np.random.choice(
            ["evening", "late_night", "morning"], n_c2, p=[0.35, 0.35, 0.30]),
        "device_type": np.random.choice(["mobile"], n_c2, p=[1.0]),
        "used_recommendations": np.random.choice([0, 1], n_c2, p=[0.55, 0.45]),
    }
    data_c2["navigation_speed"] = data_c2["pieces_started"] / data_c2["total_time_min"]
    data_c2["completion_depth_avg"] = np.clip(data_c2["completion_depth_avg"], 25, 75)
    data_c2["total_time_min"] = np.clip(data_c2["total_time_min"], 5, 20)
    all_data.append(pd.DataFrame(data_c2))

    # ---- CLUSTER 3: DOWNLOAD & DEPART ----
    n_c3 = cluster_sizes[3]
    data_c3 = {
        "total_time_min": np.random.normal(5, 2, n_c3),
        "pieces_started": np.random.randint(1, 4, n_c3),
        "completion_depth_avg": np.random.normal(8, 5, n_c3),
        "complementary_interactions": np.random.randint(4, 10, n_c3),
        "thematic_diversity": np.random.randint(1, 3, n_c3),
        "pause_count": np.random.randint(0, 1, n_c3),
        "moment_of_activity": np.random.choice(
            ["morning", "afternoon", "evening", "late_night"],
            n_c3, p=[0.25, 0.25, 0.30, 0.20]),
        "device_type": np.random.choice(
            ["mobile", "tablet", "desktop"], n_c3, p=[0.50, 0.30, 0.20]),
        "used_recommendations": np.random.choice([0, 1], n_c3, p=[0.60, 0.40]),
    }
    data_c3["navigation_speed"] = data_c3["pieces_started"] / data_c3["total_time_min"]
    data_c3["completion_depth_avg"] = np.clip(data_c3["completion_depth_avg"], 2, 20)
    data_c3["total_time_min"] = np.clip(data_c3["total_time_min"], 1.5, 10)
    all_data.append(pd.DataFrame(data_c3))

    # Combine, shuffle, assign IDs, and round numeric fields
    df = pd.concat(all_data, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "session_id", [f"S{i+1:04d}" for i in range(len(df))])
    df["total_time_min"] = df["total_time_min"].round(1)
    df["completion_depth_avg"] = df["completion_depth_avg"].round(1)
    df["navigation_speed"] = df["navigation_speed"].round(3)
    return df


# =============================================================================
# DATA QUALITY AND VALIDATION
# =============================================================================
def validate_and_summarize(df):
    """Run data quality checks and print summary statistics."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing if missing.sum() > 0 else "  No missing values detected.")

    print("\nData types:")
    print(df.dtypes.to_string())
    print(f"\nDuplicate session_ids: {df['session_id'].duplicated().sum()}")

    print("\n--- Categorical Distributions ---")
    for col in ["moment_of_activity", "device_type", "used_recommendations"]:
        print(f"\n  {col}:")
        print(f"  {df[col].value_counts().to_string()}")

    print("\n--- Numeric Summary Statistics ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().round(2).to_string())
    return df[numeric_cols].describe()


# =============================================================================
# SAVE DATA AND DOCUMENTATION
# =============================================================================
def save_data(df):
    """Save the dataset, the data dictionary, and the cluster definitions."""
    csv_path = "data/raw/digital_content_sessions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Data saved: {csv_path}")

    parquet_path = "data/raw/digital_content_sessions.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  Data saved: {parquet_path}")
    except Exception as e:
        print(f"  [WARN] Parquet not saved (install pyarrow): {e}")

    data_dict = pd.DataFrame({
        "variable_name": [
            "session_id", "total_time_min", "pieces_started",
            "completion_depth_avg", "complementary_interactions",
            "thematic_diversity", "pause_count", "navigation_speed",
            "moment_of_activity", "device_type", "used_recommendations",
        ],
        "description": [
            "Unique anonymous identifier for each session",
            "Total session duration in minutes",
            "Number of content pieces started during the session",
            "Average completion percentage across all started pieces (0-100)",
            "Count of interactions with complementary features (save, bookmark, download)",
            "Number of distinct content categories visited",
            "Number of pause events during the session",
            "Calculated: pieces_started divided by total_time_min",
            "Predominant time block of activity (morning/afternoon/evening/late_night)",
            "Device category used for the session (desktop/mobile/tablet)",
            "Binary indicator: session used personalized recommendations (1=yes, 0=no)",
        ],
        "data_type": [
            "string", "float64", "int64", "float64", "int64",
            "int64", "int64", "float64", "category", "category", "int64",
        ],
    })
    data_dict.to_csv("data/raw/data_dictionary.csv", index=False)
    print("  Data dictionary saved: data/raw/data_dictionary.csv")

    cluster_doc = pd.DataFrame({
        "cluster_name": [
            "Marathon Finishers", "Sampling Nomads",
            "Burst Consumers", "Download & Depart",
        ],
        "expected_characteristics": [
            "Long sessions, few pieces, very high completion depth (80-100%), high complementary interactions, low thematic diversity, desktop-dominant, daytime hours, high recommendation usage",
            "Moderate-length sessions, many pieces (10-20), very low completion depth (4-25%), very high thematic diversity, high navigation speed, frequent pauses, mobile-dominant, afternoon/evening",
            "Short sessions (5-20 min), moderate pieces, moderate completion depth (25-75%), moderate diversity, sporadic hours including late_night, exclusively mobile, moderate recommendation usage",
            "Very brief sessions (1.5-10 min), few pieces, very low completion depth (2-20%), very high complementary interactions (downloads/saves), low diversity, varied devices, all time blocks",
        ],
        "business_implication": [
            "Loyal power users ideal for premium content tiers and deep engagement features",
            "Browsers who need curation tools and personalized recommendations to increase depth",
            "On-the-go consumers suited for mobile-optimized short-form content and offline access",
            "Content collectors who may consume later; target with offline availability and queue management",
        ],
    })
    cluster_doc.to_csv("data/raw/cluster_definitions.csv", index=False)
    print("  Cluster definitions saved: data/raw/cluster_definitions.csv")


# =============================================================================
# SRC PACKAGE WRITER
# =============================================================================
def write_src_modules(root: str = "."):
    """Write all .py modules into the src/ package."""
    src = Path(root) / "src"
    src.mkdir(parents=True, exist_ok=True)

    modules = {}

    modules["__init__.py"] = '"""src package for the digital content consumption project."""\n'

    modules["config.py"] = '''"""Central configuration: paths, columns, palette, and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
RAW_CSV = DATA_RAW / "digital_content_sessions.csv"

RANDOM_STATE = 42

NUMERICAL_COLS = [
    "total_time_min", "pieces_started", "completion_depth_avg",
    "complementary_interactions", "thematic_diversity",
    "pause_count", "navigation_speed",
]
NUMERICAL_FEATURES = [
    "total_time_min", "pieces_started", "completion_depth_avg",
    "complementary_interactions", "thematic_diversity", "pause_count",
]
CATEGORICAL_COLS = ["moment_of_activity", "device_type", "used_recommendations"]
CATEGORICAL_FEATURES = ["moment_of_activity", "device_type"]
BINARY_FEATURES = ["used_recommendations"]

PCA_VARIANCE_THRESHOLD = 0.80

COLORS = {
    "primary": "#2C3E50", "accent": "#E74C3C", "median": "#27AE60",
    "kde": "#3498DB", "scaled": "#27AE60", "original": "#3498DB",
}


def ensure_dirs():
    for d in [DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS / "tables", MODELS]:
        d.mkdir(parents=True, exist_ok=True)
'''

    modules["data_generation.py"] = '''"""Phase 0: Synthetic session data generation."""
import numpy as np
import pandas as pd
from src import config


def generate_synthetic_sessions(n_sessions: int = 1500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    np.random.seed(seed)
    proportions = [0.25, 0.30, 0.25, 0.20]
    sizes = [int(n_sessions * p) for p in proportions]
    sizes[-1] = n_sessions - sum(sizes[:-1])
    frames = []

    n = sizes[0]
    c0 = {
        "total_time_min": np.random.normal(52, 9, n),
        "pieces_started": np.random.randint(2, 5, n),
        "completion_depth_avg": np.random.normal(91, 5, n),
        "complementary_interactions": np.random.randint(6, 14, n),
        "thematic_diversity": np.random.randint(1, 3, n),
        "pause_count": np.random.randint(0, 1, n),
        "moment_of_activity": np.random.choice(["morning", "afternoon"], n, p=[0.5, 0.5]),
        "device_type": np.random.choice(["desktop", "tablet"], n, p=[0.9, 0.1]),
        "used_recommendations": np.random.choice([0, 1], n, p=[0.1, 0.9]),
    }
    c0["navigation_speed"] = c0["pieces_started"] / c0["total_time_min"]
    c0["completion_depth_avg"] = np.clip(c0["completion_depth_avg"], 80, 100)
    c0["total_time_min"] = np.clip(c0["total_time_min"], 35, 80)
    frames.append(pd.DataFrame(c0))

    n = sizes[1]
    c1 = {
        "total_time_min": np.random.normal(22, 5, n),
        "pieces_started": np.random.randint(10, 20, n),
        "completion_depth_avg": np.random.normal(12, 5, n),
        "complementary_interactions": np.random.randint(0, 2, n),
        "thematic_diversity": np.random.randint(5, 10, n),
        "pause_count": np.random.randint(1, 5, n),
        "moment_of_activity": np.random.choice(["afternoon", "evening"], n, p=[0.4, 0.6]),
        "device_type": np.random.choice(["mobile", "tablet"], n, p=[0.85, 0.15]),
        "used_recommendations": np.random.choice([0, 1], n, p=[0.75, 0.25]),
    }
    c1["navigation_speed"] = c1["pieces_started"] / c1["total_time_min"]
    c1["completion_depth_avg"] = np.clip(c1["completion_depth_avg"], 4, 25)
    c1["total_time_min"] = np.clip(c1["total_time_min"], 10, 36)
    frames.append(pd.DataFrame(c1))

    n = sizes[2]
    c2 = {
        "total_time_min": np.random.normal(11, 3, n),
        "pieces_started": np.random.randint(3, 7, n),
        "completion_depth_avg": np.random.normal(48, 12, n),
        "complementary_interactions": np.random.randint(1, 4, n),
        "thematic_diversity": np.random.randint(1, 4, n),
        "pause_count": np.random.randint(0, 2, n),
        "moment_of_activity": np.random.choice(
            ["evening", "late_night", "morning"], n, p=[0.35, 0.35, 0.30]),
        "device_type": np.random.choice(["mobile"], n, p=[1.0]),
        "used_recommendations": np.random.choice([0, 1], n, p=[0.55, 0.45]),
    }
    c2["navigation_speed"] = c2["pieces_started"] / c2["total_time_min"]
    c2["completion_depth_avg"] = np.clip(c2["completion_depth_avg"], 25, 75)
    c2["total_time_min"] = np.clip(c2["total_time_min"], 5, 20)
    frames.append(pd.DataFrame(c2))

    n = sizes[3]
    c3 = {
        "total_time_min": np.random.normal(5, 2, n),
        "pieces_started": np.random.randint(1, 4, n),
        "completion_depth_avg": np.random.normal(8, 5, n),
        "complementary_interactions": np.random.randint(4, 10, n),
        "thematic_diversity": np.random.randint(1, 3, n),
        "pause_count": np.random.randint(0, 1, n),
        "moment_of_activity": np.random.choice(
            ["morning", "afternoon", "evening", "late_night"], n, p=[0.25, 0.25, 0.30, 0.20]),
        "device_type": np.random.choice(
            ["mobile", "tablet", "desktop"], n, p=[0.50, 0.30, 0.20]),
        "used_recommendations": np.random.choice([0, 1], n, p=[0.60, 0.40]),
    }
    c3["navigation_speed"] = c3["pieces_started"] / c3["total_time_min"]
    c3["completion_depth_avg"] = np.clip(c3["completion_depth_avg"], 2, 20)
    c3["total_time_min"] = np.clip(c3["total_time_min"], 1.5, 10)
    frames.append(pd.DataFrame(c3))

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "session_id", [f"S{i+1:04d}" for i in range(len(df))])
    df["total_time_min"] = df["total_time_min"].round(1)
    df["completion_depth_avg"] = df["completion_depth_avg"].round(1)
    df["navigation_speed"] = df["navigation_speed"].round(3)
    return df


def save_raw(df: pd.DataFrame) -> None:
    config.ensure_dirs()
    df.to_csv(config.RAW_CSV, index=False)
    try:
        df.to_parquet(config.DATA_RAW / "digital_content_sessions.parquet", index=False)
    except Exception as e:
        print(f"  [WARN] Parquet not saved: {e}")
    print(f"  Data saved: {config.RAW_CSV}")


def run(n_sessions: int = 1500) -> pd.DataFrame:
    df = generate_synthetic_sessions(n_sessions)
    save_raw(df)
    return df
'''

    modules["eda.py"] = '''"""Phase 1: Exploratory data analysis (statistics, outliers, correlation)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config


def distribution_stats(data: pd.Series) -> dict:
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    return {"mean": data.mean(), "median": data.median(), "std": data.std(),
            "min": data.min(), "max": data.max(), "skewness": data.skew(),
            "kurtosis": data.kurtosis(), "Q1": q1, "Q3": q3, "IQR": iqr,
            "outlier_lower": q1 - 1.5 * iqr, "outlier_upper": q3 + 1.5 * iqr}


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in config.NUMERICAL_COLS:
        s = distribution_stats(df[col].dropna())
        data = df[col].dropna()
        mask = (data < s["outlier_lower"]) | (data > s["outlier_upper"])
        pct = mask.sum() / len(data)
        sev = "HIGH" if pct > 0.10 else "MEDIUM" if pct > 0.05 else "LOW"
        records.append({"Feature": col, "Outlier_Count": int(mask.sum()),
                        "Outlier_Pct": round(pct * 100, 2), "Severity": sev})
    return pd.DataFrame(records)


def correlation_heatmap(df: pd.DataFrame, save: bool = True):
    corr = df[config.NUMERICAL_COLS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Spearman Correlation Matrix")
    if save:
        config.ensure_dirs()
        fig.savefig(config.FIGURES / "correlation_heatmap.png", dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return corr


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    sns.set_theme(style="whitegrid", palette="viridis")
    if df is None:
        df = pd.read_csv(config.RAW_CSV)
    out = outlier_summary(df)
    config.ensure_dirs()
    out.to_csv(config.REPORTS / "outlier_analysis_summary.csv", index=False)
    correlation_heatmap(df)
    print("  Phase 1 (EDA) completed.")
    return out
'''

    modules["preprocessing.py"] = '''"""Phase 2: Scaling + one-hot encoding -> feature matrix."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import config


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(config.RAW_CSV)

    session_ids = df["session_id"].copy()
    scaler = StandardScaler()
    X_num = pd.DataFrame(
        scaler.fit_transform(df[config.NUMERICAL_FEATURES]),
        columns=[f"{c}_scaled" for c in config.NUMERICAL_FEATURES],
        index=df.index,
    )
    X_cat = pd.get_dummies(df[config.CATEGORICAL_FEATURES],
                           columns=config.CATEGORICAL_FEATURES, drop_first=False, dtype=int)
    X_bin = df[config.BINARY_FEATURES].reset_index(drop=True)

    X_final = pd.concat([X_num, X_cat.reset_index(drop=True), X_bin], axis=1)
    X_final.insert(0, "session_id", session_ids)

    config.ensure_dirs()
    X_final.to_csv(config.DATA_PROCESSED / "sessions_preprocessed.csv", index=False)
    pd.DataFrame({"feature": config.NUMERICAL_FEATURES,
                  "mean": scaler.mean_, "std": scaler.scale_}
                 ).to_csv(config.DATA_PROCESSED / "scaler_parameters.csv", index=False)
    print(f"  Phase 2 completed: {X_final.shape[0]} rows x {X_final.shape[1]-1} features.")
    return X_final
'''

    modules["dimensionality_reduction.py"] = '''"""Phase 3: PCA (>=80% variance) + t-SNE."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src import config


def run(df_processed: pd.DataFrame = None) -> pd.DataFrame:
    if df_processed is None:
        df_processed = pd.read_csv(config.DATA_PROCESSED / "sessions_preprocessed.csv")

    session_ids = df_processed["session_id"].copy()
    X = df_processed.drop(columns=["session_id"])

    pca_full = PCA(random_state=config.RANDOM_STATE).fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= config.PCA_VARIANCE_THRESHOLD) + 1)

    pca = PCA(n_components=n_comp, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_comp)])
    df_pca.insert(0, "session_id", session_ids)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=config.RANDOM_STATE, init="pca", learning_rate="auto")
    df_tsne = pd.DataFrame(tsne.fit_transform(X), columns=["tSNE_1", "tSNE_2"])
    df_tsne.insert(0, "session_id", session_ids)

    config.ensure_dirs()
    df_pca.to_csv(config.DATA_PROCESSED / "sessions_pca.csv", index=False)
    df_tsne.to_csv(config.DATA_PROCESSED / "sessions_tsne.csv", index=False)
    pd.DataFrame(pca.components_.T,
                 columns=[f"PC{i+1}" for i in range(n_comp)],
                 index=X.columns).to_csv(config.DATA_PROCESSED / "pca_loadings.csv")
    print(f"  Phase 3 completed: {X.shape[1]} -> {n_comp} components ({cum[n_comp-1]*100:.1f}%).")
    return df_pca
'''

    modules["clustering.py"] = '''"""Phase 4: K-Means (selection by silhouette score)."""
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src import config


def find_best_k(X, k_min=2, k_max=10):
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=25, random_state=config.RANDOM_STATE).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score


def run(df_pca: pd.DataFrame = None) -> pd.DataFrame:
    if df_pca is None:
        df_pca = pd.read_csv(config.DATA_PROCESSED / "sessions_pca.csv")

    session_ids = df_pca["session_id"].copy()
    pcs = [c for c in df_pca.columns if c.startswith("PC")]
    X = df_pca[pcs].values

    best_k, _ = find_best_k(X)
    model = KMeans(n_clusters=best_k, n_init=25, random_state=config.RANDOM_STATE)
    labels = model.fit_predict(X)

    print(f"  K-Means k={best_k} | silhouette={silhouette_score(X, labels):.4f} "
          f"| davies-bouldin={davies_bouldin_score(X, labels):.4f}")

    results = pd.DataFrame({"session_id": session_ids, "kmeans_cluster": labels})
    config.ensure_dirs()
    results.to_csv(config.DATA_PROCESSED / "cluster_assignments.csv", index=False)
    joblib.dump(model, config.MODELS / "kmeans_model.pkl")
    print("  Phase 4 completed: cluster_assignments.csv + kmeans_model.pkl")
    return results
'''

    modules["profiling.py"] = '''"""Phase 5: Cluster profiling and persona mapping."""
import pandas as pd
from src import config

PERSONA_MAP = {0: "Download & Depart", 1: "Sampling Nomads",
               2: "Marathon Finishers", 3: "Burst Consumers"}
STRATEGY_MAP = {
    0: "Offline access, queue management, push notifications",
    1: "Curated discovery feeds, topic bundles, social sharing",
    2: "Premium long-form content, sequential playlists, loyalty",
    3: "Short-form mobile UX, seamless resume, micro-playlists",
}


def run() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV).merge(
        pd.read_csv(config.DATA_PROCESSED / "cluster_assignments.csv"), on="session_id")

    profile = df.groupby("kmeans_cluster")[config.NUMERICAL_FEATURES].mean().round(2)
    config.ensure_dirs()
    profile.to_csv(config.REPORTS / "tables" / "cluster_numeric_profile.csv")

    rows = []
    for c in sorted(df["kmeans_cluster"].unique()):
        size = int((df["kmeans_cluster"] == c).sum())
        rows.append({"Cluster": c, "Persona": PERSONA_MAP.get(c, f"Cluster {c}"),
                     "Share": f"{size/len(df)*100:.1f}%", "Count": size,
                     "Strategy": STRATEGY_MAP.get(c, "")})
    personas = pd.DataFrame(rows)
    personas.to_csv(config.REPORTS / "cluster_persona_mapping.csv", index=False)
    print("  Phase 5 completed: cluster_persona_mapping.csv")
    return personas
'''

    modules["pipeline.py"] = '''"""Run the full end-to-end pipeline."""
from src import config, data_generation, eda, preprocessing
from src import dimensionality_reduction, clustering, profiling


def main(n_sessions: int = 1500):
    config.ensure_dirs()
    print("[0/5] Generating data...");         df = data_generation.run(n_sessions)
    print("[1/5] EDA...");                      eda.run(df)
    print("[2/5] Preprocessing...");            df_proc = preprocessing.run(df)
    print("[3/5] Dimensionality reduction..."); df_pca = dimensionality_reduction.run(df_proc)
    print("[4/5] Clustering...");               clustering.run(df_pca)
    print("[5/5] Profiling...");                profiling.run()
    print("\\n  Pipeline complete.")


if __name__ == "__main__":
    main()
'''

    for fname, content in modules.items():
        (src / fname).write_text(content, encoding="utf-8")
        print(f"  Written: src/{fname}")
    return True


# =============================================================================
# NOTEBOOK WRITER (01..03)
# =============================================================================
def _nb(cells):
    """Build an nbformat v4 notebook from a list of cells."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


_BOOTSTRAP = '''import sys, os
sys.path.append(os.path.abspath(".."))  # add project root to path
'''


def write_notebooks(root: str = "."):
    """Write notebooks/01..03 that orchestrate the src/ pipeline phases."""
    nb_dir = Path(root) / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    nb01 = _nb([
        _md("# 01 - Exploratory Data Analysis (EDA)\n"
            "Generates the data (if missing) and runs Phase 1."),
        _code(_BOOTSTRAP),
        _code("from src import config, data_generation, eda\n"
              "import pandas as pd\n"
              "config.ensure_dirs()"),
        _md("## Generate / load data"),
        _code("if not config.RAW_CSV.exists():\n"
              "    df = data_generation.run(1500)\n"
              "else:\n"
              "    df = pd.read_csv(config.RAW_CSV)\n"
              "df.head()"),
        _md("## Outlier summary + correlation heatmap"),
        _code("outliers = eda.run(df)\n"
              "outliers"),
        _md("The correlation figure is saved to `reports/figures/correlation_heatmap.png`."),
    ])

    nb02 = _nb([
        _md("# 02 - Clustering Modeling\n"
            "Preprocessing, dimensionality reduction (PCA/t-SNE), and K-Means."),
        _code(_BOOTSTRAP),
        _code("from src import config, preprocessing, dimensionality_reduction, clustering\n"
              "config.ensure_dirs()"),
        _md("## Phase 2 - Preprocessing"),
        _code("df_proc = preprocessing.run()\n"
              "df_proc.head()"),
        _md("## Phase 3 - Dimensionality reduction (PCA + t-SNE)"),
        _code("df_pca = dimensionality_reduction.run(df_proc)\n"
              "df_pca.head()"),
        _md("## Phase 4 - K-Means"),
        _code("assignments = clustering.run(df_pca)\n"
              "assignments['kmeans_cluster'].value_counts()"),
    ])

    nb03 = _nb([
        _md("# 03 - Evaluation & Insights\n"
            "Cluster profiling and mapping to business personas."),
        _code(_BOOTSTRAP),
        _code("from src import config, profiling\n"
              "import pandas as pd\n"
              "config.ensure_dirs()"),
        _md("## Phase 5 - Profiling + personas"),
        _code("personas = profiling.run()\n"
              "personas"),
        _md("## Numeric profile per cluster"),
        _code("pd.read_csv(config.REPORTS / 'tables' / 'cluster_numeric_profile.csv')"),
        _md("Outputs saved to `reports/cluster_persona_mapping.csv` "
            "and `reports/tables/cluster_numeric_profile.csv`."),
    ])

    for name, nb in [("01_exploratory_analysis.ipynb", nb01),
                     ("02_clustering_modeling.ipynb", nb02),
                     ("03_evaluation_insights.ipynb", nb03)]:
        (nb_dir / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  Written: notebooks/{name}")
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("DIGITAL CONTENT CONSUMPTION PATTERNS")
    print("Synthetic Data Generator - Unsupervised Learning")
    print("=" * 60)

    # [1/6] Create folder structure
    print("\n[1/6] Creating project directory structure...")
    create_project_structure()

    # [2/6] Populate src/
    print("\n[2/6] Writing modules into src/ ...")
    write_src_modules(".")

    # [3/6] Generate notebooks/
    print("\n[3/6] Writing notebooks/ ...")
    write_notebooks(".")

    # [4/6] Generate synthetic data
    print("\n[4/6] Generating 1,500 synthetic sessions...")
    df_sessions = generate_synthetic_sessions(n_sessions=1500, seed=42)

    # [5/6] Validate
    print("\n[5/6] Validating generated data...")
    validate_and_summarize(df_sessions)

    # [6/6] Save data and documentation
    print("\n[6/6] Saving data and documentation...")
    save_data(df_sessions)

    # Preview
    print("\n" + "=" * 60)
    print("DATA PREVIEW (First 10 rows)")
    print("=" * 60)
    print(df_sessions.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"Project root: {os.getcwd()}/")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - src/ -> 9 modules (.py)")
    print("  - notebooks/ -> 01, 02, 03 (.ipynb)")
    print("  - data/raw/ -> CSV, Parquet, dictionary, definitions")
