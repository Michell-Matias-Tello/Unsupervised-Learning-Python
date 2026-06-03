"""Phase 5: Cluster profiling and persona mapping."""
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
