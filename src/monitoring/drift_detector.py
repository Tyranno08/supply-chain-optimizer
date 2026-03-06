# src/monitoring/drift_detector.py
# ============================================
# DATA DRIFT DETECTION
# Compares current feature distributions against
# the training baseline using statistical tests.
# Triggers retraining alert if drift is detected.
# ============================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("DriftDetector")

# ---- Paths ----
FEATURE_MATRIX_PATH  = "data/processed/feature_matrix.csv"
BASELINE_STATS_PATH  = "models/baseline_feature_stats.json"
DRIFT_REPORT_PATH    = "data/processed/drift_report.json"

# ---- Thresholds ----
KS_P_VALUE_THRESHOLD    = 0.05   # p < 0.05 = statistically significant drift
PSI_THRESHOLD           = 0.20   # PSI > 0.2 = major distribution shift
DRIFT_FEATURE_THRESHOLD = 0.20   # Flag if >20% of features drift


# ============================================
# UTILITY: Population Stability Index (PSI)
# ============================================
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Computes PSI between expected (baseline) and actual (current) distributions.
    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Moderate change — monitor
    PSI > 0.2  : Major shift — retrain
    """
    # Create equal-width bins from expected distribution
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        buckets + 1
    )

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid log(0) errors
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 6))


# ============================================
# STEP 1: Build Baseline Stats from Training Data
# ============================================
def build_baseline_stats(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Computes and saves baseline statistics from the training feature matrix.
    Run this ONCE after initial training to establish the reference distribution.
    """
    logger.info("Building baseline feature statistics...")
    stats_dict = {}

    for col in feature_cols:
        series = df[col].dropna().values
        stats_dict[col] = {
            "mean":   float(np.mean(series)),
            "std":    float(np.std(series)),
            "min":    float(np.min(series)),
            "max":    float(np.max(series)),
            "p25":    float(np.percentile(series, 25)),
            "p50":    float(np.percentile(series, 50)),
            "p75":    float(np.percentile(series, 75)),
            "sample": series[:500].tolist()   # Store 500 samples for KS test
        }

    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(stats_dict, f, indent=2)

    logger.info(f"Baseline stats saved → {BASELINE_STATS_PATH}")
    logger.info(f"  Features tracked: {len(stats_dict)}")
    return stats_dict


# ============================================
# STEP 2: Run Drift Detection
# ============================================
def run_drift_detection() -> dict:
    """
    Compares current feature matrix against the saved baseline.
    Uses KS test + PSI to detect distribution drift.
    Returns a drift report dictionary.
    """
    logger.info("=" * 55)
    logger.info("  DATA DRIFT DETECTION — STARTING")
    logger.info("=" * 55)

    # ---- Load current feature matrix ----
    if not os.path.exists(FEATURE_MATRIX_PATH):
        raise FileNotFoundError(f"Feature matrix not found: {FEATURE_MATRIX_PATH}")

    df = pd.read_csv(FEATURE_MATRIX_PATH)
    logger.info(f"Current feature matrix loaded — Shape: {df.shape}")

    # ---- Load or build baseline ----
    if not os.path.exists(BASELINE_STATS_PATH):
        logger.warning("No baseline found. Building baseline from current data...")
        feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]
        build_baseline_stats(df, feature_cols)
        logger.info("Baseline built. Run again tomorrow to detect actual drift.")
        return {"drift_detected": False, "note": "Baseline just created."}

    with open(BASELINE_STATS_PATH) as f:
        baseline = json.load(f)

    logger.info(f"Baseline loaded — {len(baseline)} features tracked.")

    # ---- Run tests per feature ----
    drift_results  = {}
    drifted_features = []
    stable_features  = []

    for col, base_stats in baseline.items():
        if col not in df.columns:
            continue

        current_vals  = df[col].dropna().values
        baseline_vals = np.array(base_stats["sample"])

        if len(current_vals) < 30 or len(baseline_vals) < 30:
            continue

        # KS Test
        ks_stat, ks_p = stats.ks_2samp(baseline_vals, current_vals)

        # PSI
        psi = compute_psi(baseline_vals, current_vals)

        # Mean shift (z-score based)
        mean_shift_z = abs(
            (np.mean(current_vals) - base_stats["mean"])
            / max(base_stats["std"], 1e-6)
        )

        is_drifted = (ks_p < KS_P_VALUE_THRESHOLD) or (psi > PSI_THRESHOLD)

        drift_results[col] = {
            "ks_statistic":  round(float(ks_stat), 6),
            "ks_p_value":    round(float(ks_p), 6),
            "psi":           round(psi, 6),
            "mean_shift_z":  round(float(mean_shift_z), 4),
            "baseline_mean": round(base_stats["mean"], 4),
            "current_mean":  round(float(np.mean(current_vals)), 4),
            "drifted":       is_drifted
        }

        if is_drifted:
            drifted_features.append(col)
        else:
            stable_features.append(col)

    # ---- Compute overall drift score ----
    total_features  = len(drift_results)
    n_drifted       = len(drifted_features)
    drift_pct       = n_drifted / max(total_features, 1)
    drift_detected  = drift_pct > DRIFT_FEATURE_THRESHOLD

    # ---- Build report ----
    report = {
        "timestamp":         datetime.now().isoformat(),
        "drift_detected":    drift_detected,
        "drift_pct":         round(drift_pct, 4),
        "n_features_total":  total_features,
        "n_features_drifted": n_drifted,
        "n_features_stable": len(stable_features),
        "drifted_features":  drifted_features,
        "stable_features":   stable_features[:10],  # top 10 for brevity
        "feature_details":   drift_results,
        "thresholds": {
            "ks_p_value":         KS_P_VALUE_THRESHOLD,
            "psi":                PSI_THRESHOLD,
            "feature_drift_pct":  DRIFT_FEATURE_THRESHOLD
        }
    }

    # ---- Save report ----
    os.makedirs("data/processed", exist_ok=True)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    # ---- Print summary ----
    logger.info("=" * 55)
    logger.info("  DRIFT DETECTION RESULTS")
    logger.info("=" * 55)
    logger.info(f"  Total features tested:  {total_features}")
    logger.info(f"  Features drifted:       {n_drifted} ({drift_pct:.1%})")
    logger.info(f"  Features stable:        {len(stable_features)}")
    logger.info(f"  Drift threshold:        >{DRIFT_FEATURE_THRESHOLD:.0%} of features")
    logger.info("-" * 55)

    if drift_detected:
        logger.info("  *** DRIFT DETECTED *** — Retraining recommended!")
        logger.info("  Top drifted features:")
        for feat in drifted_features[:5]:
            d = drift_results[feat]
            logger.info(
                f"    {feat:<35} | KS p={d['ks_p_value']:.4f} | PSI={d['psi']:.4f}"
            )
    else:
        logger.info("  No significant drift detected — Model is stable.")

    logger.info(f"  Drift report saved → {DRIFT_REPORT_PATH}")
    logger.info("=" * 55)

    return report


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    report = run_drift_detection()

    if report.get("drift_detected"):
        print("DRIFT DETECTED")
        sys.exit(0)
    else:
        print("NO DRIFT")
        sys.exit(0)
