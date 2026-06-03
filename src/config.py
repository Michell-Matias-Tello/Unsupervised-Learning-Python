"""Central configuration: paths, columns, palette, and constants."""
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
