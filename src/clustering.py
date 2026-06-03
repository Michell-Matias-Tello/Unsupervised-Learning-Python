"""Phase 4: K-Means (selection by silhouette score)."""
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
