"""Phase 1: Exploratory data analysis (statistics, outliers, correlation)."""
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
