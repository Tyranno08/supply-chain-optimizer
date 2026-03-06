# src/models/train_xgboost.py
# ============================================
# XGBOOST BASELINE MODEL
# Trains two XGBoost models:
#   1. Classifier  → P(delay > 2 hours)
#   2. Regressor   → Predicted delay in hours
# Uses the feature matrix from Phase 4
# ============================================
'''
import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("XGBoostTrainer")

# ============================================
# PATHS
# ============================================
FEATURE_MATRIX_PATH     = "data/processed/feature_matrix.csv"
XGBOOST_CLF_PATH        = "models/xgboost_classifier.joblib"
XGBOOST_REG_PATH        = "models/xgboost_regressor.joblib"
SCALER_PATH             = "models/feature_scaler.joblib"

os.makedirs("models", exist_ok=True)

# ============================================
# FEATURE COLUMNS
# (Must match exactly what Phase 4 produced)
# ============================================
FEATURE_COLS = [
    # Temporal
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end",
    "is_peak_season", "rolling_7d_avg_delay",
    "rolling_30d_avg_delay", "rolling_7d_std_delay",
    "lag_1_delay", "lag_2_delay",

    # Route / Logistics
    "standard_duration_hours", "standard_cost",
    "transport_mode_enc", "cargo_type_enc",
    "route_id_enc", "cargo_weight_tons",

    # Geospatial
    "distance_km", "calculated_distance_km",
    "expected_speed_kmh", "source_hemisphere",
    "dest_hemisphere", "crosses_hemisphere",
    "crosses_pacific", "crosses_atlantic",
    "capacity_ratio", "lon_diff", "lat_diff",
    "source_capacity", "dest_capacity",

    # NLP Risk
    "source_avg_risk", "source_max_risk",
    "source_negative_events", "source_event_count",
    "dest_avg_risk", "dest_max_risk",
    "dest_negative_events", "dest_event_count",
    "combined_route_risk",

    # Network Centrality
    "source_betweenness", "source_degree_cent",
    "source_closeness", "source_in_degree", "source_out_degree",
    "dest_betweenness", "dest_degree_cent",
    "dest_closeness", "dest_in_degree", "dest_out_degree",
]

TARGET_CLF = "delay_flag"           # Binary classification target
TARGET_REG = "actual_delay_hours"   # Regression target


class XGBoostTrainer:

    def __init__(self):
        self.clf_model  = None
        self.reg_model  = None
        self.scaler     = StandardScaler()
        self.feature_cols_used = []

    # ============================================
    # DATA LOADING & PREPARATION
    # ============================================
    def load_data(self) -> tuple:
        """
        Loads the feature matrix, selects available
        feature columns, and splits into X and y.
        """
        logger.info(f"Loading feature matrix from {FEATURE_MATRIX_PATH}...")

        df = pd.read_csv(FEATURE_MATRIX_PATH)
        logger.info(f"Raw data shape: {df.shape}")

        # Use only columns that actually exist in the CSV
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]

        if missing:
            logger.warning(f"Missing feature columns (will be skipped): {missing}")

        self.feature_cols_used = available
        logger.info(f"Using {len(available)} feature columns.")

        # Drop rows where target is NaN
        df = df.dropna(subset=[TARGET_CLF, TARGET_REG])
        df[TARGET_CLF] = df[TARGET_CLF].astype(int)

        X = df[available].fillna(0).values
        y_clf = df[TARGET_CLF].values
        y_reg = df[TARGET_REG].values

        logger.info(
            f"Dataset ready — Rows: {len(X)}, "
            f"Features: {X.shape[1]}, "
            f"Delay rate: {y_clf.mean():.2%}"
        )

        return X, y_clf, y_reg

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> tuple:
        """
        Time-aware train/test split.
        We use the LAST 20% of data as test set
        to simulate real production conditions
        (you never test on the past).
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            f"Train: {len(X_train)} rows, "
            f"Test: {len(X_test)} rows"
        )
        return X_train, X_test, y_train, y_test

    # ============================================
    # TRAIN CLASSIFIER
    # ============================================
    def train_classifier(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray
    ) -> None:
        """
        Trains XGBoost binary classifier with:
        - Proper class weight calculation
        - Probability calibration (fixes AUC issue)
        - Threshold optimization
        """
        from sklearn.calibration import CalibratedClassifierCV

        logger.info("Training XGBoost Classifier (Delay Prediction)...")

        # ---- Compute class weight from actual data ----
        # This is better than SMOTE for AUC improvement
        n_negative  = (y_train == 0).sum()
        n_positive  = (y_train == 1).sum()
        scale_pos   = n_negative / max(n_positive, 1)

        logger.info(
            f"Class distribution — On-Time: {n_negative}, "
            f"Delayed: {n_positive}, "
            f"scale_pos_weight: {scale_pos:.2f}"
        )

        # ---- Scale features ----
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # ---- Base XGBoost model ----
        base_clf = xgb.XGBClassifier(
            n_estimators      = 400,
            max_depth         = 5,
            learning_rate     = 0.03,
            subsample         = 0.75,
            colsample_bytree  = 0.75,
            min_child_weight  = 5,
            gamma             = 0.2,
            reg_alpha         = 0.3,
            reg_lambda        = 2.0,
            scale_pos_weight  = scale_pos,   # Uses real class ratio
            use_label_encoder = False,
            eval_metric       = "auc",       # Optimize for AUC not logloss
            random_state      = 42,
            n_jobs            = -1
        )

        # ---- Probability Calibration ----
        # This is the KEY fix for the low AUC problem
        # Isotonic calibration reshapes probability outputs so they
        # better reflect true delay probabilities
        self.clf_model = CalibratedClassifierCV(
            base_clf,
            method = "isotonic",   # Better than sigmoid for XGBoost
            cv     = 3
        )

        # ---- Cross-validation BEFORE calibration ----
        logger.info("Running 5-Fold Stratified Cross-Validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_clf,
            X_train_scaled,
            y_train,
            cv      = cv,
            scoring = "roc_auc",   # Use AUC for CV scoring — more honest
            n_jobs  = -1
        )
        logger.info(
            f"CV AUC Scores: {cv_scores.round(4)} | "
            f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # ---- Train calibrated model ----
        self.clf_model.fit(X_train_scaled, y_train)

        # ---- Find optimal threshold using Youden's J ----
        y_pred_prob = self.clf_model.predict_proba(X_test_scaled)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        youden_j     = tpr - fpr
        optimal_idx  = np.argmax(youden_j)
        optimal_thresh = thresholds[optimal_idx]

        logger.info(f"Optimal decision threshold: {optimal_thresh:.4f}")

        # Use optimal threshold for final predictions
        y_pred = (y_pred_prob >= optimal_thresh).astype(int)

        f1      = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_prob)

        logger.info("=" * 55)
        logger.info("CALIBRATED CLASSIFIER RESULTS")
        logger.info("=" * 55)
        logger.info(f"F1 Score:        {f1:.4f}")
        logger.info(f"AUC-ROC:         {auc_roc:.4f}")
        logger.info(f"Optimal Thresh:  {optimal_thresh:.4f}")
        logger.info("\nClassification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=["On Time", "Delayed"]
        )
        for line in report.split("\n"):
            logger.info(line)
        logger.info("=" * 55)

        self._log_feature_importance(base_clf, top_n=15)

        # Save calibrated model, threshold, and scaler
        joblib.dump(self.scaler,     SCALER_PATH)
        joblib.dump(self.clf_model,  XGBOOST_CLF_PATH)

        # Save optimal threshold separately for use in API
        threshold_data = {"optimal_threshold": float(optimal_thresh)}
        import json
        with open("models/clf_threshold.json", "w") as f:
            json.dump(threshold_data, f)

        logger.info(f"Calibrated classifier saved → {XGBOOST_CLF_PATH}")
        logger.info(f"Optimal threshold saved    → models/clf_threshold.json")

    # ============================================
    # TRAIN REGRESSOR
    # ============================================
    def train_regressor(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray
    ) -> None:
        """
        Trains XGBoost regressor with:
        - Log-transform of target (fixes skewed delay hours)
        - Huber loss (robust to outliers in delay data)
        """
        logger.info("Training XGBoost Regressor (Delay Hours Prediction)...")

        # ---- Log-transform the target ----
        # actual_delay_hours is right-skewed (many zeros, few large values)
        # log1p(x) = log(x+1) transforms this to a more normal distribution
        # We inverse-transform predictions back to hours at inference time
        y_train_log = np.log1p(y_train)
        y_test_log  = np.log1p(y_test)

        logger.info(
            f"Target after log1p transform — "
            f"Mean: {y_train_log.mean():.4f}, "
            f"Std: {y_train_log.std():.4f}, "
            f"Max: {y_train_log.max():.4f}"
        )

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.reg_model = xgb.XGBRegressor(
            n_estimators     = 400,
            max_depth        = 5,
            learning_rate    = 0.03,
            subsample        = 0.75,
            colsample_bytree = 0.75,
            min_child_weight = 5,
            reg_alpha        = 0.3,
            reg_lambda       = 2.0,
            objective        = "reg:pseudohubererror",  # Robust to outliers
            random_state     = 42,
            n_jobs           = -1
        )

        self.reg_model.fit(
            X_train_scaled,
            y_train_log,       # Train on log-transformed target
            eval_set  = [(X_test_scaled, y_test_log)],
            verbose   = False
        )

        # ---- Evaluate in original scale ----
        y_pred_log   = self.reg_model.predict(X_test_scaled)
        y_pred_hours = np.expm1(y_pred_log)   # Inverse of log1p

        mae  = mean_absolute_error(y_test, y_pred_hours)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_hours))
        r2   = r2_score(y_test, y_pred_hours)

        # Safe MAPE — exclude near-zero actuals to avoid division explosion
        nonzero_mask = y_test > 1.0
        if nonzero_mask.sum() > 0:
            mape = np.mean(
                np.abs(
                    (y_test[nonzero_mask] - y_pred_hours[nonzero_mask]) /
                    y_test[nonzero_mask]
                )
            ) * 100
        else:
            mape = 0.0

        logger.info("=" * 55)
        logger.info("REGRESSOR RESULTS (original scale)")
        logger.info("=" * 55)
        logger.info(f"MAE  (hours):   {mae:.4f}")
        logger.info(f"RMSE (hours):   {rmse:.4f}")
        logger.info(f"R²   Score:     {r2:.4f}")
        logger.info(f"MAPE (%):       {mape:.4f}%  (non-zero actuals only)")
        logger.info("=" * 55)

        # Save model with log-transform flag in metadata
        joblib.dump(self.reg_model, XGBOOST_REG_PATH)
        logger.info(f"Regressor saved → {XGBOOST_REG_PATH}")

        # Save transform info for inference
        import json
        transform_info = {
            "target_transform": "log1p",
            "inverse_transform": "expm1"
        }
        with open("models/reg_transform.json", "w") as f:
            json.dump(transform_info, f)
        logger.info("Regressor transform metadata saved → models/reg_transform.json")

    # ============================================
    # FEATURE IMPORTANCE LOGGING
    # ============================================
    def _log_feature_importance(
        self,
        model,
        top_n: int = 15
    ) -> None:
        """Logs the top N most important features."""        
        importance = model.feature_importances_
        feat_imp   = sorted(
            zip(self.feature_cols_used, importance),
            key     = lambda x: x[1],
            reverse = True
        )[:top_n]

        logger.info(f"TOP {top_n} FEATURE IMPORTANCES:")
        for feat, imp in feat_imp:
            bar = "█" * int(imp * 50)
            logger.info(f"  {feat:<35} {imp:.4f}  {bar}")


def run_xgboost_training() -> None:
    """Orchestrates the full XGBoost training pipeline."""
    logger.info("=" * 55)
    logger.info("  XGBOOST TRAINING PIPELINE — STARTING")
    logger.info("=" * 55)

    trainer = XGBoostTrainer()

    # Load data
    X, y_clf, y_reg = trainer.load_data()

    # Split for classifier (uses delay_flag)
    X_tr_c, X_te_c, y_tr_c, y_te_c = trainer.split_data(X, y_clf)

    # Split for regressor (uses actual_delay_hours)
    X_tr_r, X_te_r, y_tr_r, y_te_r = trainer.split_data(X, y_reg)

    # Train both models
    trainer.train_classifier(X_tr_c, X_te_c, y_tr_c, y_te_c)
    trainer.train_regressor(X_tr_r, X_te_r, y_tr_r, y_te_r)

    logger.info("=" * 55)
    logger.info("  XGBOOST TRAINING COMPLETE")
    logger.info("=" * 55)


if __name__ == "__main__":
    run_xgboost_training()

'''

# src/models/train_xgboost.py
# ============================================
# XGBOOST BASELINE MODEL
# Trains two XGBoost models:
#   1. Classifier  → P(delay > 2 hours)
#   2. Regressor   → Predicted delay in hours
# Uses the feature matrix from Phase 4
# ============================================

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("XGBoostTrainer")

# ============================================
# PATHS
# ============================================
FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"
XGBOOST_CLF_PATH    = "models/xgboost_classifier.joblib"
XGBOOST_REG_PATH    = "models/xgboost_regressor.joblib"
SCALER_PATH         = "models/feature_scaler.joblib"

os.makedirs("models", exist_ok=True)

# ============================================
# FEATURE COLUMNS
# (Must match exactly what Phase 4 produced)
# ============================================
FEATURE_COLS = [
    # Temporal
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end",
    "is_peak_season", "rolling_7d_avg_delay",
    "rolling_30d_avg_delay", "rolling_7d_std_delay",
    "lag_1_delay", "lag_2_delay",

    # Route / Logistics
    "standard_duration_hours", "standard_cost",
    "transport_mode_enc", "cargo_type_enc",
    "route_id_enc", "cargo_weight_tons",

    # Geospatial
    "distance_km", "calculated_distance_km",
    "expected_speed_kmh", "source_hemisphere",
    "dest_hemisphere", "crosses_hemisphere",
    "crosses_pacific", "crosses_atlantic",
    "capacity_ratio", "lon_diff", "lat_diff",
    "source_capacity", "dest_capacity",

    # NLP Risk
    "source_avg_risk", "source_max_risk",
    "source_negative_events", "source_event_count",
    "dest_avg_risk", "dest_max_risk",
    "dest_negative_events", "dest_event_count",
    "combined_route_risk",

    # Network Centrality
    "source_betweenness", "source_degree_cent",
    "source_closeness", "source_in_degree", "source_out_degree",
    "dest_betweenness", "dest_degree_cent",
    "dest_closeness", "dest_in_degree", "dest_out_degree",
]

TARGET_CLF = "delay_flag"           # Binary classification target
TARGET_REG = "actual_delay_hours"   # Regression target


class XGBoostTrainer:

    def __init__(self):
        self.clf_model         = None
        self.reg_model         = None
        self.base_clf          = None   # Keep reference to base (unfitted) for feature importance
        self.scaler            = StandardScaler()
        self.feature_cols_used = []

    # ============================================
    # DATA LOADING & PREPARATION
    # ============================================
    def load_data(self) -> tuple:
        """
        Loads the feature matrix, selects available
        feature columns, and splits into X and y.
        """
        logger.info(f"Loading feature matrix from {FEATURE_MATRIX_PATH}...")

        df = pd.read_csv(FEATURE_MATRIX_PATH)
        logger.info(f"Raw data shape: {df.shape}")

        # Use only columns that actually exist in the CSV
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]

        if missing:
            logger.warning(f"Missing feature columns (will be skipped): {missing}")

        self.feature_cols_used = available
        logger.info(f"Using {len(available)} feature columns.")

        # Drop rows where target is NaN
        df = df.dropna(subset=[TARGET_CLF, TARGET_REG])
        df[TARGET_CLF] = df[TARGET_CLF].astype(int)

        X     = df[available].fillna(0).values
        y_clf = df[TARGET_CLF].values
        # Use continuous engineered target if available
        # Falls back to original if not present
        if "actual_delay_hours_continuous" in df.columns:
            y_reg = df["actual_delay_hours_continuous"].values
            logger.info(
                "Using continuous engineered delay target for regression."
            )
        else:
            y_reg = df[TARGET_REG].values
            logger.info(
                "Using original delay hours target for regression."
            )

        logger.info(
            f"Dataset ready — Rows: {len(X)}, "
            f"Features: {X.shape[1]}, "
            f"Delay rate: {y_clf.mean():.2%}"
        )

        return X, y_clf, y_reg

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> tuple:
        """
        Time-aware train/test split.
        We use the LAST 20% of data as test set
        to simulate real production conditions
        (you never test on the past).
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            f"Train: {len(X_train)} rows, "
            f"Test: {len(X_test)} rows"
        )
        return X_train, X_test, y_train, y_test

    # ============================================
    # TRAIN CLASSIFIER
    # ============================================
    def train_classifier(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray
    ) -> None:
        """
        Trains XGBoost binary classifier with:
        - Proper class weight calculation
        - Probability calibration (fixes AUC issue)
        - Threshold optimization
        """
        logger.info("Training XGBoost Classifier (Delay Prediction)...")

        # ---- Compute class weight from actual data ----
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos  = n_negative / max(n_positive, 1)

        logger.info(
            f"Class distribution — On-Time: {n_negative}, "
            f"Delayed: {n_positive}, "
            f"scale_pos_weight: {scale_pos:.2f}"
        )

        # ---- Scale features ----
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # ---- Base XGBoost model ----
        self.base_clf = xgb.XGBClassifier(
            n_estimators      = 400,
            max_depth         = 5,
            learning_rate     = 0.03,
            subsample         = 0.75,
            colsample_bytree  = 0.75,
            min_child_weight  = 5,
            gamma             = 0.2,
            reg_alpha         = 0.3,
            reg_lambda        = 2.0,
            scale_pos_weight  = scale_pos,
            use_label_encoder = False,
            eval_metric       = "auc",
            random_state      = 42,
            n_jobs            = -1
        )

        # ---- Probability Calibration ----
        self.clf_model = CalibratedClassifierCV(
            self.base_clf,
            method = "isotonic",
            cv     = 3
        )

        # ---- Cross-validation on base clf BEFORE calibration ----
        logger.info("Running 5-Fold Stratified Cross-Validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.base_clf,
            X_train_scaled,
            y_train,
            cv      = cv,
            scoring = "roc_auc",
            n_jobs  = -1
        )
        logger.info(
            f"CV AUC Scores: {cv_scores.round(4)} | "
            f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # ---- Train calibrated model (this also fits base_clf internally) ----
        self.clf_model.fit(X_train_scaled, y_train)

        # ---- Find optimal threshold using Youden's J ----
        y_pred_prob  = self.clf_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        youden_j       = tpr - fpr
        optimal_idx    = np.argmax(youden_j)
        optimal_thresh = thresholds[optimal_idx]

        logger.info(f"Optimal decision threshold: {optimal_thresh:.4f}")

        # Use optimal threshold for final predictions
        y_pred  = (y_pred_prob >= optimal_thresh).astype(int)
        f1      = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_prob)

        logger.info("=" * 55)
        logger.info("CALIBRATED CLASSIFIER RESULTS")
        logger.info("=" * 55)
        logger.info(f"F1 Score:        {f1:.4f}")
        logger.info(f"AUC-ROC:         {auc_roc:.4f}")
        logger.info(f"Optimal Thresh:  {optimal_thresh:.4f}")
        logger.info("\nClassification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=["On Time", "Delayed"]
        )
        for line in report.split("\n"):
            logger.info(line)
        logger.info("=" * 55)

        # ---- Feature importance from the calibrated model ----
        self._log_feature_importance(self.clf_model, top_n=15)

        # ---- Save calibrated model, threshold, and scaler ----
        joblib.dump(self.scaler,    SCALER_PATH)
        joblib.dump(self.clf_model, XGBOOST_CLF_PATH)

        threshold_data = {"optimal_threshold": float(optimal_thresh)}
        with open("models/clf_threshold.json", "w") as f:
            json.dump(threshold_data, f)

        logger.info(f"Calibrated classifier saved → {XGBOOST_CLF_PATH}")
        logger.info(f"Optimal threshold saved    → models/clf_threshold.json")

    # ============================================
    # TRAIN REGRESSOR
    # ============================================
    def train_regressor(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray
    ) -> None:
        """
        Trains XGBoost regressor with:
        - Log-transform of target (fixes skewed delay hours)
        - Huber loss (robust to outliers in delay data)
        """
        logger.info("Training XGBoost Regressor (Delay Hours Prediction)...")

        # ---- Log-transform the target ----
        y_train_log = np.log1p(y_train)
        y_test_log  = np.log1p(y_test)

        logger.info(
            f"Target after log1p transform — "
            f"Mean: {y_train_log.mean():.4f}, "
            f"Std: {y_train_log.std():.4f}, "
            f"Max: {y_train_log.max():.4f}"
        )

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.reg_model = xgb.XGBRegressor(
            n_estimators     = 400,
            max_depth        = 5,
            learning_rate    = 0.03,
            subsample        = 0.75,
            colsample_bytree = 0.75,
            min_child_weight = 5,
            reg_alpha        = 0.3,
            reg_lambda       = 2.0,
            objective        = "reg:pseudohubererror",
            random_state     = 42,
            n_jobs           = -1
        )

        self.reg_model.fit(
            X_train_scaled,
            y_train_log,
            eval_set = [(X_test_scaled, y_test_log)],
            verbose  = False
        )

        # ---- Evaluate in original scale ----
        y_pred_log   = self.reg_model.predict(X_test_scaled)
        y_pred_hours = np.expm1(y_pred_log)

        mae  = mean_absolute_error(y_test, y_pred_hours)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_hours))
        r2   = r2_score(y_test, y_pred_hours)

        # Safe MAPE — exclude near-zero actuals to avoid division explosion
        nonzero_mask = y_test > 1.0
        if nonzero_mask.sum() > 0:
            mape = np.mean(
                np.abs(
                    (y_test[nonzero_mask] - y_pred_hours[nonzero_mask]) /
                    y_test[nonzero_mask]
                )
            ) * 100
        else:
            mape = 0.0

        logger.info("=" * 55)
        logger.info("REGRESSOR RESULTS (original scale)")
        logger.info("=" * 55)
        logger.info(f"MAE  (hours):   {mae:.4f}")
        logger.info(f"RMSE (hours):   {rmse:.4f}")
        logger.info(f"R2   Score:     {r2:.4f}")
        logger.info(f"MAPE (%):       {mape:.4f}%  (non-zero actuals only)")
        logger.info("=" * 55)

        joblib.dump(self.reg_model, XGBOOST_REG_PATH)
        logger.info(f"Regressor saved → {XGBOOST_REG_PATH}")

        transform_info = {
            "target_transform":   "log1p",
            "inverse_transform":  "expm1"
        }
        with open("models/reg_transform.json", "w") as f:
            json.dump(transform_info, f)
        logger.info("Regressor transform metadata saved → models/reg_transform.json")

    # ============================================
    # FEATURE IMPORTANCE LOGGING
    # ============================================
    def _log_feature_importance(
        self,
        model,
        top_n: int = 15
    ) -> None:
        """Logs the top N most important features."""
        # Handle CalibratedClassifierCV wrapper
        if hasattr(model, 'calibrated_classifiers_'):
            base       = model.calibrated_classifiers_[0].estimator
            importance = base.feature_importances_
        else:
            importance = model.feature_importances_

        feat_imp = sorted(
            zip(self.feature_cols_used, importance),
            key     = lambda x: x[1],
            reverse = True
        )[:top_n]

        logger.info(f"TOP {top_n} FEATURE IMPORTANCES:")
        for feat, imp in feat_imp:
            bar = "#" * int(imp * 50)   # Using # instead of unicode block chars
            logger.info(f"  {feat:<35} {imp:.4f}  {bar}")

# ============================================
# DELAY SEVERITY CLASSIFIER
# Replaces the regression model with an
# ordinal multi-class classifier
# ============================================

SEVERITY_CLF_PATH = "models/xgboost_severity_classifier.joblib"

SEVERITY_LABELS = {
    0: "On Time",
    1: "Minor Delay (1-24h)",
    2: "Major Delay (24-48h)",
    3: "Severe Delay (48h+)"
}

class DelaySeverityClassifier:
    """
    4-class ordinal classifier that predicts
    HOW SEVERE a delay will be, not just
    whether it will happen.

    This is more actionable for operations teams
    and more trainable given day-granularity data.
    """

    def __init__(self):
        self.model  = None
        self.scaler = joblib.load(SCALER_PATH)

    def load_data(self) -> tuple:
        """Loads feature matrix with severity target."""
        logger.info("Loading data for severity classifier...")

        df = pd.read_csv(FEATURE_MATRIX_PATH)

        if "delay_severity" not in df.columns:
            logger.error(
                "delay_severity column not found. "
                "Re-run feature pipeline first."
            )
            raise ValueError("delay_severity column missing from feature matrix.")

        available = [c for c in FEATURE_COLS if c in df.columns]
        df        = df.dropna(subset=["delay_severity"])
        df["delay_severity"] = df["delay_severity"].astype(int)

        X = df[available].fillna(0).values
        y = df["delay_severity"].values

        logger.info(
            f"Severity dataset — Rows: {len(X)}, "
            f"Classes: {np.unique(y)}"
        )

        # Class distribution
        for cls in sorted(np.unique(y)):
            count = (y == cls).sum()
            pct   = count / len(y) * 100
            logger.info(
                f"  Class {cls} [{SEVERITY_LABELS[cls]}]: "
                f"{count:,} ({pct:.1f}%)"
            )

        return X, y, available

    def train(self) -> None:
        """
        Trains XGBoost multi-class severity classifier
        with class weight balancing.
        """
        from sklearn.metrics import classification_report

        logger.info("Training Delay Severity Classifier...")

        X, y, feature_names = self.load_data()

        # Time-aware split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_sc = self.scaler.transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        self.model = xgb.XGBClassifier(
            n_estimators     = 400,
            max_depth        = 5,
            learning_rate    = 0.03,
            subsample        = 0.75,
            colsample_bytree = 0.75,
            min_child_weight = 3,
            reg_alpha        = 0.3,
            reg_lambda       = 2.0,
            objective        = "multi:softprob",
            num_class        = 4,
            eval_metric      = "mlogloss",
            random_state     = 42,
            n_jobs           = -1
        )

        self.model.fit(
            X_train_sc,
            y_train,
            eval_set = [(X_test_sc, y_test)],
            verbose  = False
        )

        # ---- Evaluation ----
        y_pred      = self.model.predict(X_test_sc)
        y_pred_prob = self.model.predict_proba(X_test_sc)

        # Per-class metrics
        report = classification_report(
            y_test, y_pred,
            target_names = list(SEVERITY_LABELS.values()),
            digits       = 4
        )

        # Weighted F1
        weighted_f1 = f1_score(
            y_test, y_pred,
            average="weighted"
        )

        # Ordinal accuracy — predictions within 1 class count as correct
        ordinal_acc = np.mean(np.abs(y_test - y_pred) <= 1)

        logger.info("=" * 65)
        logger.info("  DELAY SEVERITY CLASSIFIER RESULTS")
        logger.info("=" * 65)
        logger.info(f"  Weighted F1 Score:     {weighted_f1:.4f}")
        logger.info(f"  Ordinal Accuracy:      {ordinal_acc:.4f}")
        logger.info(f"  (Ordinal = within 1 severity class)")
        logger.info("\nPer-Class Report:")
        for line in report.split("\n"):
            if line.strip():
                logger.info(f"  {line}")
        logger.info("=" * 65)

        # Save model
        joblib.dump(self.model, SEVERITY_CLF_PATH)
        logger.info(f"Severity classifier saved → {SEVERITY_CLF_PATH}")

        # Save metrics for evaluation report
        import json
        severity_metrics = {
            "weighted_f1":    round(weighted_f1, 4),
            "ordinal_accuracy": round(ordinal_acc, 4),
            "classes":        SEVERITY_LABELS
        }
        with open("models/severity_metrics.json", "w") as f:
            json.dump(severity_metrics, f, indent=4)
        logger.info(
            "Severity metrics saved → models/severity_metrics.json"
        )

def run_xgboost_training() -> None:
    """
    Orchestrates the full XGBoost training pipeline
    with MLflow experiment tracking.
    All parameters, metrics and artifacts are
    automatically logged for reproducibility.
    """
    import mlflow
    import mlflow.xgboost
    import json

    # ---- Configure MLflow ----
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("supply_chain_delay_prediction")

    logger.info("=" * 55)
    logger.info("  XGBOOST TRAINING PIPELINE — STARTING")
    logger.info("  MLflow tracking: mlruns/")
    logger.info("=" * 55)

    trainer = XGBoostTrainer()
    X, y_clf, y_reg = trainer.load_data()

    X_tr_c, X_te_c, y_tr_c, y_te_c = trainer.split_data(X, y_clf)
    X_tr_r, X_te_r, y_tr_r, y_te_r = trainer.split_data(X, y_reg)

    # ============================================
    # CLASSIFIER RUN
    # ============================================
    with mlflow.start_run(run_name="xgboost_classifier") as clf_run:

        # Log parameters
        mlflow.log_params({
            "model_type":        "XGBClassifier_Calibrated",
            "n_estimators":      400,
            "max_depth":         5,
            "learning_rate":     0.03,
            "subsample":         0.75,
            "colsample_bytree":  0.75,
            "calibration_method": "isotonic",
            "cv_folds":          5,
            "test_size":         0.2,
            "train_samples":     len(X_tr_c),
            "test_samples":      len(X_te_c),
            "n_features":        X_tr_c.shape[1],
            "delay_rate_train":  round(float(y_tr_c.mean()), 4),
            "dataset":           "DataCo_Kaggle_2016_2018",
            "target":            "delay_flag"
        })

        # Train
        trainer.train_classifier(X_tr_c, X_te_c, y_tr_c, y_te_c)

        # Log metrics
        from sklearn.metrics import (
            f1_score, roc_auc_score, accuracy_score,
            precision_score, recall_score
        )
        X_te_scaled = trainer.scaler.transform(X_te_c)
        y_pred      = trainer.clf_model.predict(X_te_scaled)
        y_prob      = trainer.clf_model.predict_proba(X_te_scaled)[:, 1]

        mlflow.log_metrics({
            "accuracy":  round(float(accuracy_score(y_te_c, y_pred)),             4),
            "f1_score":  round(float(f1_score(y_te_c, y_pred)),                   4),
            "precision": round(float(precision_score(y_te_c, y_pred,
                                                      zero_division=0)),           4),
            "recall":    round(float(recall_score(y_te_c, y_pred,
                                                  zero_division=0)),               4),
            "auc_roc":   round(float(roc_auc_score(y_te_c, y_prob)),              4)
        })

        # Log model artifact
        mlflow.sklearn.log_model(
            trainer.clf_model,
            artifact_path   = "classifier",
            registered_model_name = "supply_chain_delay_classifier"
        )

        # Log scaler
        mlflow.sklearn.log_model(
            trainer.scaler,
            artifact_path = "scaler"
        )

        # Log threshold file
        mlflow.log_artifact("models/clf_threshold.json")

        logger.info(f"Classifier run logged to MLflow. Run ID: {clf_run.info.run_id}")

    # ============================================
    # REGRESSOR RUN
    # ============================================
    with mlflow.start_run(run_name="xgboost_regressor") as reg_run:

        mlflow.log_params({
            "model_type":       "XGBRegressor_LogTransform",
            "n_estimators":     400,
            "max_depth":        5,
            "learning_rate":    0.03,
            "objective":        "reg:pseudohubererror",
            "target_transform": "log1p",
            "test_size":        0.2,
            "train_samples":    len(X_tr_r),
            "test_samples":     len(X_te_r),
            "dataset":          "DataCo_Kaggle_2016_2018",
            "target":           "actual_delay_hours_continuous"
        })

        trainer.train_regressor(X_tr_r, X_te_r, y_tr_r, y_te_r)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        X_te_scaled  = trainer.scaler.transform(X_te_r)
        y_pred_log   = trainer.reg_model.predict(X_te_scaled)
        y_pred_hours = np.expm1(y_pred_log)

        mlflow.log_metrics({
            "mae":  round(float(mean_absolute_error(y_te_r, y_pred_hours)),                     4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te_r, y_pred_hours))),             4),
            "r2":   round(float(r2_score(y_te_r, y_pred_hours)),                               4)
        })

        mlflow.xgboost.log_model(
            trainer.reg_model,
            artifact_path         = "regressor",
            registered_model_name = "supply_chain_delay_regressor"
        )

        logger.info(f"Regressor run logged to MLflow. Run ID: {reg_run.info.run_id}")

    # ============================================
    # SEVERITY CLASSIFIER RUN
    # ============================================
    with mlflow.start_run(run_name="severity_classifier") as sev_run:

        mlflow.log_params({
            "model_type":    "XGBClassifier_MultiClass",
            "n_classes":     4,
            "objective":     "multi:softprob",
            "n_estimators":  400,
            "dataset":       "DataCo_Kaggle_2016_2018",
            "target":        "delay_severity"
        })

        severity_clf = DelaySeverityClassifier()
        severity_clf.train()

        if os.path.exists("models/severity_metrics.json"):
            with open("models/severity_metrics.json") as f:
                sev_metrics = json.load(f)
            mlflow.log_metrics({
                "weighted_f1":      sev_metrics.get("weighted_f1",     0),
                "ordinal_accuracy": sev_metrics.get("ordinal_accuracy", 0)
            })

        mlflow.sklearn.log_model(
            severity_clf.model,
            artifact_path         = "severity_classifier",
            registered_model_name = "supply_chain_severity_classifier"
        )

        logger.info(f"Severity classifier run logged. Run ID: {sev_run.info.run_id}")

    logger.info("=" * 55)
    logger.info("  XGBOOST TRAINING COMPLETE")
    logger.info("  View experiments: mlflow ui")
    logger.info("=" * 55)

if __name__ == "__main__":
    run_xgboost_training()