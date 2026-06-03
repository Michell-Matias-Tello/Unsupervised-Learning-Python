"""Run the full end-to-end pipeline."""
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
    print("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
