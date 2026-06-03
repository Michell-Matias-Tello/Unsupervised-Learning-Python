"""Phase 0: Synthetic session data generation."""
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
