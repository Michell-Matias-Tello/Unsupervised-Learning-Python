"""Phase 3: PCA (>=80% variance) + t-SNE."""
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
