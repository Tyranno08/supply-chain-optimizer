# src/monitoring/build_baseline.py
# ============================================
# BASELINE BUILDER
# Run this ONCE after initial model training
# to establish the reference distribution for
# drift detection comparisons.
# ============================================

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from src.monitoring.drift_detector import build_baseline_stats
from src.utils import get_logger

logger = get_logger("BaselineBuilder")

FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"

# Features to track for drift (numeric only)
NUMERIC_FEATURES = [
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end", "is_peak_season",
    "rolling_7d_avg_delay", "rolling_30d_avg_delay", "rolling_7d_std_delay",
    "lag_1_delay", "lag_2_delay",
    "standard_duration_hours", "standard_cost",
    "cargo_weight_tons", "distance_km",
    "source_avg_risk", "source_max_risk", "dest_avg_risk", "dest_max_risk",
    "combined_route_risk", "source_betweenness", "dest_betweenness",
    "capacity_ratio", "lon_diff", "lat_diff"
]

if __name__ == "__main__":
    logger.info("Loading feature matrix for baseline creation...")

    if not os.path.exists(FEATURE_MATRIX_PATH):
        logger.error(f"Feature matrix not found at {FEATURE_MATRIX_PATH}")
        sys.exit(1)

    df = pd.read_csv(FEATURE_MATRIX_PATH)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Filter to only existing numeric columns
    valid_features = [f for f in NUMERIC_FEATURES if f in df.columns]
    logger.info(f"Tracking {len(valid_features)} features for drift detection.")

    stats = build_baseline_stats(df, valid_features)

    logger.info("=" * 50)
    logger.info("  BASELINE CREATION COMPLETE")
    logger.info(f"  Features tracked: {len(stats)}")
    logger.info("  Saved → models/baseline_feature_stats.json")
    logger.info("  Run drift_detector.py daily to detect shifts.")
    logger.info("=" * 50)
