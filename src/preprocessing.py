"""Phase 2: Scaling + one-hot encoding -> feature matrix."""
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
