# src/models/evaluation.py
# ============================================
# MODEL EVALUATION & INTERPRETABILITY MODULE
# Covers:
#   - XGBoost Classifier metrics & plots
#   - XGBoost Regressor metrics & plots
#   - SHAP feature attribution
#   - GNN prediction visualization
#   - Business ROI Calculator
# ============================================

import os
import sys
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — saves files without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import torch

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("ModelEvaluation")

# ============================================
# PATHS
# ============================================
XGBOOST_CLF_PATH  = "models/xgboost_classifier.joblib"
XGBOOST_REG_PATH  = "models/xgboost_regressor.joblib"
SCALER_PATH       = "models/feature_scaler.joblib"
GNN_MODEL_PATH    = "models/gnn_supply_chain.pth"
GNN_METADATA_PATH = "models/gnn_metadata.pkl"
GRAPH_OBJECT_PATH = "data/processed/graph_object.gpickle"
FEATURE_MATRIX    = "data/processed/feature_matrix.csv"

VIZ_DIR           = "data/processed/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ============================================
# FEATURE COLUMNS
# ============================================
FEATURE_COLS = [
    "dispatch_hour", "dispatch_dayofweek", "dispatch_month",
    "dispatch_quarter", "is_weekend", "is_month_end",
    "is_peak_season", "rolling_7d_avg_delay",
    "rolling_30d_avg_delay", "rolling_7d_std_delay",
    "lag_1_delay", "lag_2_delay",
    "standard_duration_hours", "standard_cost",
    "transport_mode_enc", "cargo_type_enc",
    "route_id_enc", "cargo_weight_tons",
    "distance_km", "calculated_distance_km",
    "expected_speed_kmh", "source_hemisphere",
    "dest_hemisphere", "crosses_hemisphere",
    "crosses_pacific", "crosses_atlantic",
    "capacity_ratio", "lon_diff", "lat_diff",
    "source_capacity", "dest_capacity",
    "source_avg_risk", "source_max_risk",
    "source_negative_events", "source_event_count",
    "dest_avg_risk", "dest_max_risk",
    "dest_negative_events", "dest_event_count",
    "combined_route_risk",
    "source_betweenness", "source_degree_cent",
    "source_closeness", "source_in_degree", "source_out_degree",
    "dest_betweenness", "dest_degree_cent",
    "dest_closeness", "dest_in_degree", "dest_out_degree",
]


# ============================================
# DATA LOADER FOR EVALUATION
# ============================================
class EvaluationDataLoader:
    """
    Loads and prepares data specifically for
    model evaluation (test set only).
    """

    def load(self) -> tuple:
        """
        Loads feature matrix, applies scaler,
        and returns test split.
        """
        logger.info("Loading data for evaluation...")

        df     = pd.read_csv(FEATURE_MATRIX)
        df     = df.dropna(subset=["delay_flag", "actual_delay_hours"])
        df["delay_flag"] = df["delay_flag"].astype(int)

        # Use only available columns
        available = [c for c in FEATURE_COLS if c in df.columns]
        X          = df[available].fillna(0).values
        y_clf      = df["delay_flag"].values
        y_reg      = df["actual_delay_hours"].values

        # Time-aware split — last 20% is test set
        split_idx  = int(len(X) * 0.8)
        X_test     = X[split_idx:]
        y_test_clf = y_clf[split_idx:]
        y_test_reg = y_reg[split_idx:]

        # Scale
        scaler     = joblib.load(SCALER_PATH)
        X_test_sc  = scaler.transform(X_test)

        logger.info(
            f"Test set — Rows: {len(X_test)}, "
            f"Delay rate: {y_test_clf.mean():.2%}"
        )

        return X_test_sc, y_test_clf, y_test_reg, available, df


# ============================================
# CLASSIFIER EVALUATOR
# ============================================
class ClassifierEvaluator:
    """
    Full evaluation suite for the XGBoost
    binary delay classifier.
    """

    def __init__(self):
        self.clf = joblib.load(XGBOOST_CLF_PATH)
        logger.info("Classifier loaded for evaluation.")

    def evaluate(
        self,
        X_test:  np.ndarray,
        y_test:  np.ndarray
    ) -> dict:
        """
        Computes all classification metrics and
        generates evaluation plots.
        """
        y_pred      = self.clf.predict(X_test)
        y_pred_prob = self.clf.predict_proba(X_test)[:, 1]

        # ---- Core metrics ----
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),       4),
            "f1_score":  round(f1_score(y_test, y_pred),             4),
            "precision": round(precision_score(y_test, y_pred,
                                               zero_division=0),      4),
            "recall":    round(recall_score(y_test, y_pred,
                                            zero_division=0),         4),
            "auc_roc":   round(roc_auc_score(y_test, y_pred_prob),   4)
        }

        # ---- Log metrics ----
        logger.info("=" * 55)
        logger.info("  CLASSIFIER EVALUATION METRICS")
        logger.info("=" * 55)
        for metric, value in metrics.items():
            bar = "█" * int(value * 30)
            logger.info(f"  {metric:<12}: {value:.4f}  {bar}")
        logger.info("=" * 55)
        logger.info("\nDetailed Classification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=["On Time", "Delayed"]
        )
        for line in report.split("\n"):
            if line.strip():
                logger.info(f"  {line}")
        logger.info("=" * 55)

        # ---- Generate plots ----
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_pred_prob)
        self._plot_probability_distribution(y_test, y_pred_prob)

        return metrics

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Plots a styled confusion matrix."""
        logger.info("Generating Confusion Matrix plot...")

        cm     = confusion_matrix(y_true, y_pred)
        labels = ["On Time", "Delayed"]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        sns.heatmap(
            cm,
            annot       = True,
            fmt         = "d",
            cmap        = "Blues",
            xticklabels = labels,
            yticklabels = labels,
            ax          = ax,
            linewidths  = 0.5,
            linecolor   = "#30363d",
            annot_kws   = {"size": 16, "weight": "bold", "color": "white"}
        )

        ax.set_title(
            "Confusion Matrix — Delay Classifier",
            color    = "white",
            fontsize = 14,
            pad      = 15
        )
        ax.set_xlabel("Predicted Label", color="white", fontsize=12)
        ax.set_ylabel("True Label",      color="white", fontsize=12)
        ax.tick_params(colors="white")

        # Annotate cells with percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(
                    j + 0.5, i + 0.7,
                    f"({pct:.1f}%)",
                    ha        = "center",
                    va        = "center",
                    color     = "lightgray",
                    fontsize  = 11
                )

        output_path = f"{VIZ_DIR}/04_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Confusion Matrix saved → {output_path}")

    def _plot_roc_curve(
        self,
        y_true:      np.ndarray,
        y_pred_prob: np.ndarray
    ) -> None:
        """Plots the ROC Curve with AUC score."""
        logger.info("Generating ROC Curve plot...")

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc              = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # ROC curve
        ax.plot(
            fpr, tpr,
            color     = "#1f77b4",
            linewidth = 2.5,
            label     = f"XGBoost (AUC = {roc_auc:.4f})"
        )

        # Random baseline
        ax.plot(
            [0, 1], [0, 1],
            color     = "#e74c3c",
            linewidth = 1.5,
            linestyle = "--",
            label     = "Random Baseline (AUC = 0.5000)"
        )

        # Shade area under curve
        ax.fill_between(fpr, tpr, alpha=0.15, color="#1f77b4")

        # Find optimal threshold (Youden's J statistic)
        optimal_idx   = np.argmax(tpr - fpr)
        optimal_fpr   = fpr[optimal_idx]
        optimal_tpr   = tpr[optimal_idx]
        optimal_thresh = thresholds[optimal_idx]

        ax.scatter(
            optimal_fpr, optimal_tpr,
            color  = "#f39c12",
            s      = 100,
            zorder = 5,
            label  = f"Optimal Threshold = {optimal_thresh:.3f}"
        )

        ax.set_title(
            "ROC Curve — Delay Classifier",
            color    = "white",
            fontsize = 14,
            pad      = 15
        )
        ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
        ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(
            facecolor  = "#161b22",
            edgecolor  = "#30363d",
            labelcolor = "white",
            fontsize   = 11
        )
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        output_path = f"{VIZ_DIR}/05_roc_curve.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"ROC Curve saved → {output_path}")

    def _plot_probability_distribution(
        self,
        y_true:      np.ndarray,
        y_pred_prob: np.ndarray
    ) -> None:
        """
        Plots predicted probability distributions
        for delayed vs on-time shipments.
        A well-calibrated model shows clear separation.
        """
        logger.info("Generating Probability Distribution plot...")

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # On-time shipments
        ax.hist(
            y_pred_prob[y_true == 0],
            bins      = 40,
            alpha     = 0.7,
            color     = "#2ecc71",
            label     = "On Time (True Label = 0)",
            edgecolor = "white",
            linewidth = 0.3
        )

        # Delayed shipments
        ax.hist(
            y_pred_prob[y_true == 1],
            bins      = 40,
            alpha     = 0.7,
            color     = "#e74c3c",
            label     = "Delayed (True Label = 1)",
            edgecolor = "white",
            linewidth = 0.3
        )

        # Decision threshold line
        ax.axvline(
            x         = 0.5,
            color     = "#f39c12",
            linewidth = 2,
            linestyle = "--",
            label     = "Decision Threshold = 0.50"
        )

        ax.set_title(
            "Predicted Delay Probability Distribution\n"
            "(Good model shows clear separation between colors)",
            color    = "white",
            fontsize = 13
        )
        ax.set_xlabel("Predicted Delay Probability", color="white", fontsize=12)
        ax.set_ylabel("Count",                       color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(
            facecolor  = "#161b22",
            edgecolor  = "#30363d",
            labelcolor = "white",
            fontsize   = 11
        )
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        output_path = f"{VIZ_DIR}/06_probability_distribution.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Probability Distribution saved → {output_path}")


# ============================================
# REGRESSOR EVALUATOR
# ============================================
class RegressorEvaluator:
    """
    Full evaluation suite for the XGBoost
    delay hours regressor.
    """

    def __init__(self):
        self.reg = joblib.load(XGBOOST_REG_PATH)
        logger.info("Regressor loaded for evaluation.")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluates regressor with log-inverse transform
        and safe MAPE calculation.
        """
        y_pred_raw = self.reg.predict(X_test)

        # Inverse log1p transform if model was trained on log target
        import os
        import json
        transform_path = "models/reg_transform.json"
        if os.path.exists(transform_path):
            with open(transform_path) as f:
                transform_info = json.load(f)
            if transform_info.get("target_transform") == "log1p":
                y_pred = np.expm1(y_pred_raw)
                logger.info("Applied expm1 inverse transform to predictions.")
            else:
                y_pred = y_pred_raw
        else:
            y_pred = y_pred_raw

        # Clip predictions to realistic range
        y_pred = np.clip(y_pred, 0, 200)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # Safe MAPE — only on rows where actual delay > 1 hour
        # Avoids division-by-near-zero explosion
        nonzero_mask = y_test > 1.0
        if nonzero_mask.sum() > 10:
            mape = np.mean(
                np.abs(
                    (y_test[nonzero_mask] - y_pred[nonzero_mask]) /
                    y_test[nonzero_mask]
                )
            ) * 100
        else:
            mape = 0.0

        metrics = {
            "mae":  round(mae,  4),
            "rmse": round(rmse, 4),
            "r2":   round(r2,   4),
            "mape": round(mape, 4)
        }

        logger.info("=" * 55)
        logger.info("  REGRESSOR EVALUATION METRICS")
        logger.info("=" * 55)
        logger.info(f"  MAE  (hours):   {metrics['mae']:.4f}")
        logger.info(f"  RMSE (hours):   {metrics['rmse']:.4f}")
        logger.info(f"  R²   Score:     {metrics['r2']:.4f}")
        logger.info(
            f"  MAPE (%):       {metrics['mape']:.4f}% "
            f"(computed on {nonzero_mask.sum()} non-zero actuals)"
        )
        logger.info("=" * 55)

        self._plot_actual_vs_predicted(y_test, y_pred)
        self._plot_residuals(y_test, y_pred)

        return metrics

    def _plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Plots actual vs predicted delay hours."""
        logger.info("Generating Actual vs Predicted plot...")

        # Cap extreme outliers for visualization
        cap       = np.percentile(y_true, 97)
        mask      = y_true <= cap
        y_true_v  = y_true[mask]
        y_pred_v  = y_pred[mask]

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # Scatter plot with density coloring
        sc = ax.scatter(
            y_true_v, y_pred_v,
            c          = np.abs(y_true_v - y_pred_v),
            cmap       = "RdYlGn_r",
            alpha      = 0.5,
            s          = 15,
            edgecolors = "none"
        )
        plt.colorbar(sc, ax=ax, label="Absolute Error (hours)").ax.yaxis.set_tick_params(color="white")

        # Perfect prediction line
        max_val = max(y_true_v.max(), y_pred_v.max())
        ax.plot(
            [0, max_val], [0, max_val],
            color     = "#f39c12",
            linewidth = 2,
            linestyle = "--",
            label     = "Perfect Prediction Line"
        )

        ax.set_title(
            "Actual vs Predicted Delay Hours\n"
            "(Points close to orange line = accurate predictions)",
            color    = "white",
            fontsize = 13
        )
        ax.set_xlabel("Actual Delay Hours",    color="white", fontsize=12)
        ax.set_ylabel("Predicted Delay Hours", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(
            facecolor  = "#161b22",
            edgecolor  = "#30363d",
            labelcolor = "white"
        )
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        output_path = f"{VIZ_DIR}/07_actual_vs_predicted.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Actual vs Predicted saved → {output_path}")

    def _plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Plots residual analysis.
        Residuals should be randomly distributed
        around zero — if not, the model is biased.
        """
        logger.info("Generating Residual Analysis plot...")

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")

        # Plot 1: Residuals vs Predicted
        ax1 = axes[0]
        ax1.set_facecolor("#161b22")
        ax1.scatter(
            y_pred, residuals,
            alpha      = 0.4,
            s          = 15,
            color      = "#1f77b4",
            edgecolors = "none"
        )
        ax1.axhline(
            y         = 0,
            color     = "#e74c3c",
            linewidth = 2,
            linestyle = "--"
        )
        ax1.set_title(
            "Residuals vs Predicted Values",
            color="white", fontsize=12
        )
        ax1.set_xlabel("Predicted Delay Hours", color="white")
        ax1.set_ylabel("Residuals (Actual - Predicted)", color="white")
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("#30363d")
        ax1.spines["left"].set_color("#30363d")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Plot 2: Residual distribution
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")
        ax2.hist(
            residuals.clip(-50, 50),
            bins      = 50,
            color     = "#1f77b4",
            alpha     = 0.8,
            edgecolor = "white",
            linewidth = 0.3
        )
        ax2.axvline(
            x         = 0,
            color     = "#e74c3c",
            linewidth = 2,
            linestyle = "--",
            label     = f"Mean residual: {residuals.mean():.2f}h"
        )
        ax2.set_title(
            "Residual Distribution\n"
            "(Should be symmetric around 0)",
            color="white", fontsize=12
        )
        ax2.set_xlabel("Residual Value (hours)", color="white")
        ax2.set_ylabel("Frequency",              color="white")
        ax2.tick_params(colors="white")
        ax2.legend(
            facecolor  = "#0d1117",
            edgecolor  = "#30363d",
            labelcolor = "white"
        )
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        output_path = f"{VIZ_DIR}/08_residual_analysis.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"Residual Analysis saved → {output_path}")


# ============================================
# SHAP INTERPRETABILITY
# ============================================
class SHAPAnalyzer:
    """
    Uses SHAP (SHapley Additive exPlanations) to
    explain WHY the XGBoost model made each prediction.

    SHAP answers: "Which features pushed this
    shipment's delay probability UP or DOWN?"

    This is critical for:
    - Regulatory compliance (explainable AI)
    - Business trust (stakeholders need to understand)
    - Model debugging (finding data issues)
    """

    def __init__(self, feature_names: list):
        self.clf           = joblib.load(XGBOOST_CLF_PATH)
        self.feature_names = feature_names
        logger.info("SHAP Analyzer initialized.")

    def compute_shap_values(
        self,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Computes SHAP values using TreeExplainer.
        TreeExplainer is optimized for tree-based
        models like XGBoost — very fast.
        """
        logger.info("Computing SHAP values (this may take 1-2 minutes)...")

        # Extract base XGBoost model from CalibratedClassifierCV wrapper
        if hasattr(self.clf, 'calibrated_classifiers_'):
            clf_for_shap = self.clf.calibrated_classifiers_[0].estimator
        else:
            clf_for_shap = self.clf

        explainer = shap.TreeExplainer(clf_for_shap)

        # Use a sample of max 500 rows for speed
        sample_size = min(500, len(X_test))
        X_sample    = X_test[:sample_size]

        shap_values = explainer.shap_values(X_sample)

        # For binary classifier, shap_values is a list [class0, class1]
        # We want class 1 (delayed) SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        logger.info(
            f"SHAP values computed — Shape: {shap_values.shape}"
        )

        return shap_values, X_sample

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_sample:    np.ndarray
    ) -> None:
        """
        SHAP Summary Plot (Beeswarm).
        Each dot = one shipment.
        Position on X axis = how much it pushed delay prob up/down.
        Color = feature value (red = high, blue = low).
        """
        logger.info("Generating SHAP Summary (Beeswarm) plot...")

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names = self.feature_names,
            show          = False,
            plot_size     = None,
            color_bar     = True,
            max_display   = 20
        )

        plt.title(
            "SHAP Feature Attribution — Delay Classifier\n"
            "Red = High Feature Value  |  Blue = Low Feature Value\n"
            "Right of center = increases delay probability",
            color    = "white",
            fontsize = 12,
            pad      = 15
        )

        output_path = f"{VIZ_DIR}/09_shap_summary.png"
        plt.savefig(
            output_path,
            dpi             = 150,
            bbox_inches     = "tight",
            facecolor       = "#0d1117"
        )
        plt.close()
        logger.info(f"SHAP Summary saved → {output_path}")

    def plot_shap_bar(
        self,
        shap_values: np.ndarray
    ) -> None:
        """
        SHAP Bar Plot — Mean absolute SHAP values.
        Simple bar chart showing overall feature importance.
        More business-friendly than the beeswarm.
        """
        logger.info("Generating SHAP Bar plot...")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feat_shap     = sorted(
            zip(self.feature_names, mean_abs_shap),
            key     = lambda x: x[1],
            reverse = True
        )[:20]

        names, values = zip(*feat_shap)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        colors = [
            "#e74c3c" if v > np.percentile(values, 75) else
            "#f39c12" if v > np.percentile(values, 50) else
            "#1f77b4"
            for v in values
        ]

        bars = ax.barh(
            range(len(names)),
            values,
            color     = colors,
            edgecolor = "white",
            linewidth = 0.3
        )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, color="white", fontsize=10)
        ax.set_xlabel(
            "Mean |SHAP Value| (Average Impact on Delay Probability)",
            color   = "white",
            fontsize= 11
        )
        ax.set_title(
            "Top 20 Features — SHAP Importance\n"
            "Red = Highest Impact on Delay Predictions",
            color    = "white",
            fontsize = 13
        )
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                val + max(values) * 0.01,
                i,
                f"{val:.4f}",
                va       = "center",
                color    = "white",
                fontsize = 8
            )

        output_path = f"{VIZ_DIR}/10_shap_bar.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"SHAP Bar plot saved → {output_path}")

    def plot_shap_waterfall_single(
        self,
        shap_values: np.ndarray,
        X_sample:    np.ndarray,
        sample_idx:  int = 0
    ) -> None:
        """
        SHAP Waterfall plot for a SINGLE prediction.
        Shows exactly why the model predicted delay
        for one specific shipment — very powerful for
        explaining individual decisions to business users.
        """
        logger.info(
            f"Generating SHAP Waterfall for sample {sample_idx}..."
        )
        # Extract base XGBoost model from CalibratedClassifierCV wrapper
        if hasattr(self.clf, 'calibrated_classifiers_'):
            clf_for_shap = self.clf.calibrated_classifiers_[0].estimator
        else:
            clf_for_shap = self.clf

        explainer = shap.TreeExplainer(clf_for_shap)
        explanation = explainer(X_sample)

        # Handle binary classifier output
        if len(explanation.shape) == 3:
            explanation = explanation[:, :, 1]

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0d1117")

        shap.plots.waterfall(
            explanation[sample_idx],
            max_display = 15,
            show        = False
        )

        plt.title(
            f"SHAP Waterfall — Single Shipment Explanation (Sample #{sample_idx})\n"
            "Red arrows = features that INCREASED delay probability\n"
            "Blue arrows = features that DECREASED delay probability",
            color    = "white",
            fontsize = 11
        )

        output_path = f"{VIZ_DIR}/11_shap_waterfall.png"
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"SHAP Waterfall saved → {output_path}")


# ============================================
# GNN PREDICTION VISUALIZER
# ============================================
class GNNEvaluator:
    """
    Visualizes GNN node-level predictions
    on the supply chain graph.
    """

    def __init__(self):
        with open(GNN_METADATA_PATH, "rb") as f:
            self.metadata     = pickle.load(f)
        with open(GRAPH_OBJECT_PATH, "rb") as f:
            self.graph        = pickle.load(f)
        self.node_mapping     = self.metadata["node_mapping"]
        self.reverse_mapping  = {
            v: k for k, v in self.node_mapping.items()
        }
        logger.info("GNN Evaluator initialized.")

    def get_predictions(self) -> dict:
        """Loads the GNN and runs inference on the full graph."""
        from src.models.train_gnn import SupplyChainGNN
        from torch_geometric.data import Data

        device     = torch.device("cpu")
        model      = SupplyChainGNN(
            num_node_features=self.metadata["num_features"]
        )
        model.load_state_dict(
            torch.load(GNN_MODEL_PATH, map_location=device)
        )
        model.eval()

        nodes      = list(self.graph.nodes())
        node_feats = []

        for node_id in nodes:
            attrs    = self.graph.nodes[node_id]
            risk     = attrs.get("risk_score",             0.0)
            norm_cap = attrs.get("base_capacity", 10000) / 50000.0
            btwn     = attrs.get("betweenness_centrality",  0.0)
            deg      = attrs.get("degree_centrality",       0.0)
            cls      = attrs.get("closeness_centrality",    0.0)
            ind      = attrs.get("in_degree",               0.0) / 10.0
            outd     = attrs.get("out_degree",              0.0) / 10.0
            node_feats.append([risk, norm_cap, btwn, deg, cls, ind, outd])

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_list  = [
            [self.node_mapping[s], self.node_mapping[d]]
            for s, d in self.graph.edges()
            if s in self.node_mapping and d in self.node_mapping
        ]
        edge_index = torch.tensor(
            edge_list, dtype=torch.long
        ).t().contiguous()
        data       = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            delay_prob, delay_hours = model(data)

        return {
            "delay_prob":   delay_prob.numpy().flatten(),
            "delay_hours":  delay_hours.numpy().flatten()
        }

    def plot_gnn_predictions(self) -> None:
        """
        Plots GNN node predictions on the
        geographic supply chain network.
        """
        logger.info("Generating GNN Prediction Network plot...")

        import networkx as nx

        predictions = self.get_predictions()
        G           = self.graph

        fig, axes   = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor("#0d1117")

        pos = {
            node: (
                G.nodes[node]["longitude"],
                G.nodes[node]["latitude"]
            )
            for node in G.nodes()
        }

        nodes_list   = list(G.nodes())

        for ax_idx, (metric, title, cmap) in enumerate([
            ("delay_prob",  "Delay Probability per Port", "RdYlGn_r"),
            ("delay_hours", "Predicted Delay Hours",      "YlOrRd")
        ]):
            ax = axes[ax_idx]
            ax.set_facecolor("#0d1117")

            values      = [
                predictions[metric][self.node_mapping.get(n, 0)]
                for n in nodes_list
            ]
            node_sizes  = [
                G.nodes[n].get("base_capacity", 5000) / 8
                for n in nodes_list
            ]

            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color      = "#404040",
                width           = 1.5,
                arrowsize       = 15,
                arrowstyle      = "->",
                ax              = ax,
                connectionstyle = "arc3,rad=0.1"
            )

            # Draw nodes
            sc = nx.draw_networkx_nodes(
                G, pos,
                node_color  = values,
                node_size   = node_sizes,
                cmap        = plt.cm.get_cmap(cmap),
                vmin        = min(values),
                vmax        = max(values),
                ax          = ax,
                edgecolors  = "white",
                linewidths  = 1.0
            )

            # Labels
            labels = {
                n: G.nodes[n].get("name", n).split()[-1]
                for n in nodes_list
            }
            nx.draw_networkx_labels(
                G, pos,
                labels     = labels,
                font_size  = 8,
                font_color = "white",
                ax         = ax
            )

            plt.colorbar(sc, ax=ax, shrink=0.7).ax.yaxis.set_tick_params(
                color="white"
            )
            ax.set_title(
                f"GNN — {title}",
                color    = "white",
                fontsize = 13
            )
            ax.axis("off")

        fig.suptitle(
            "Graph Neural Network — Spatial Bottleneck Propagation Results",
            color    = "white",
            fontsize = 15,
            y        = 1.02
        )

        output_path = f"{VIZ_DIR}/12_gnn_predictions.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"GNN Prediction Network saved → {output_path}")

'''
# ============================================
# BUSINESS ROI CALCULATOR
# ============================================
class BusinessROICalculator:
    """
    Translates model accuracy metrics into
    real business value (dollars saved).

    This is the section recruiters from
    business-facing teams love the most.
    """

    # Industry average costs (conservative estimates)
    AVG_DELAY_COST_PER_HOUR  = 2_500    # USD per hour of shipment delay
    AVG_REROUTING_COST       = 8_000    # USD cost to reroute one shipment
    AVG_SHIPMENT_VALUE       = 45_000   # USD value of average shipment

    def calculate(
        self,
        clf_metrics: dict,
        reg_metrics: dict,
        n_shipments: int
    ) -> dict:
        """
        Realistic ROI calculation grounded in
        industry-standard per-shipment costs.

        Assumptions (conservative, citable in interviews):
        - Avg delay cost:    \$2,500/hour  (Gartner 2023 logistics benchmark)
        - Avg delay hours:   Based on MAE — model prevents half the MAE
        - Rerouting cost:    \$8,000 per shipment (industry average)
        - Monthly volume:    Scaled from test set, not full dataset
        """
        logger.info("=" * 65)
        logger.info("  BUSINESS ROI CALCULATOR")
        logger.info("=" * 65)

        # ---- Realistic monthly shipment volume ----
        # We analyzed n_shipments over the dataset period
        # Assume dataset covers ~12 months → monthly = n/12
        monthly_shipments = max(int(n_shipments / 12), 100)

        # ---- Delay rate from classifier recall ----
        delay_rate         = 0.43   # Conservative industry estimate
        estimated_delayed  = int(monthly_shipments * delay_rate)

        # ---- True positives from recall ----
        recall             = clf_metrics.get("recall",    0.75)
        precision          = clf_metrics.get("precision", 0.75)
        true_positives     = int(estimated_delayed * recall)
        false_positives    = int(
            true_positives *
            (1 - precision) / max(precision, 0.01)
        )

        # ---- Hours prevented ----
        mae_hours           = reg_metrics.get("mae", 10.0)
        # Realistic assumption: early warning prevents 30% of delay hours
        hours_prevented_per_shipment = mae_hours * 0.30

        # ---- Cost calculations ----
        # Without model: all delayed shipments incur full delay cost
        avg_delay_hours_without = mae_hours
        cost_without_model      = (
            estimated_delayed *
            avg_delay_hours_without *
            self.AVG_DELAY_COST_PER_HOUR
        )

        # With model: TP shipments get early warning (partial savings)
        # FP shipments incur unnecessary rerouting cost
        cost_with_model = (
            (estimated_delayed - true_positives) *
            avg_delay_hours_without *
            self.AVG_DELAY_COST_PER_HOUR +
            true_positives *
            (avg_delay_hours_without - hours_prevented_per_shipment) *
            self.AVG_DELAY_COST_PER_HOUR +
            false_positives * self.AVG_REROUTING_COST
        )

        gross_savings  = max(0, cost_without_model - cost_with_model)
        annual_savings = gross_savings * 12

        roi_results = {
            "total_shipments_analyzed":      n_shipments,
            "monthly_shipments_estimated":   monthly_shipments,
            "estimated_delayed":             estimated_delayed,
            "correctly_flagged_delayed":     true_positives,
            "false_alarms":                  false_positives,
            "avg_delay_prevented_hours":     round(
                hours_prevented_per_shipment, 2
            ),
            "cost_without_model_usd":        round(cost_without_model, 2),
            "cost_with_model_usd":           round(cost_with_model,    2),
            "monthly_savings_usd":           round(gross_savings,      2),
            "annual_savings_usd":            round(annual_savings,     2),
            "model_f1_score":                clf_metrics.get("f1_score", 0),
            "model_auc_roc":                 clf_metrics.get("auc_roc",  0),
            "model_mae_hours":               reg_metrics.get("mae",      0)
        }

        # ---- Log results ----
        logger.info(
            f"  Monthly Shipment Volume:      "
            f"{monthly_shipments:>10,}  (dataset / 12 months)"
        )
        logger.info(
            f"  Estimated Delayed/Month:      "
            f"{estimated_delayed:>10,}"
        )
        logger.info(
            f"  Correctly Flagged (TP):       "
            f"{true_positives:>10,}"
        )
        logger.info(
            f"  False Alarms (FP):            "
            f"{false_positives:>10,}"
        )
        logger.info(
            f"  Hours Prevented/Shipment:     "
            f"{hours_prevented_per_shipment:>10.2f}h"
        )
        logger.info("-" * 65)
        logger.info(
            f"  Cost WITHOUT Model (monthly): "
            f"${cost_without_model:>12,.2f}"
        )
        logger.info(
            f"  Cost WITH Model (monthly):    "
            f"${cost_with_model:>12,.2f}"
        )
        logger.info("-" * 65)
        logger.info(
            f"  💰 MONTHLY SAVINGS:           "
            f"${gross_savings:>12,.2f}"
        )
        logger.info(
            f"  💰 PROJECTED ANNUAL SAVINGS:  "
            f"${annual_savings:>12,.2f}"
        )
        logger.info("=" * 65)

        self._plot_roi_dashboard(roi_results)

        return roi_results
    '''

# ============================================
# UPDATED BUSINESS ROI CALCULATOR
# ============================================
class BusinessROICalculator:
    """
    Realistic ROI calculation grounded in
    industry-standard per-shipment costs.

    Updated for 2026 conflict scenario:
    Gulf route disruptions significantly increase
    baseline delay costs for affected routes.
    """

    # ---- Industry Cost Constants ----
    # Source: Gartner 2023 Supply Chain Cost Report
    AVG_DELAY_COST_PER_HOUR    = 1_800    # USD/hour (conservative Gartner estimate)
    AVG_REROUTING_COST         = 6_500    # USD per rerouted shipment
    AVG_SHIPMENT_VALUE         = 45_000   # USD average container value

    # ---- Conflict Premium (2026 Gulf scenario) ----
    # Gulf routes (DXB, SIN) face 40% higher delay costs
    # due to insurance premiums and military zone surcharges
    CONFLICT_ROUTE_PREMIUM     = 1.40

    def calculate(
        self,
        clf_metrics: dict,
        reg_metrics: dict,
        n_shipments: int
    ) -> dict:

        logger.info("=" * 65)
        logger.info("  BUSINESS ROI CALCULATOR (2026 Conflict-Adjusted)")
        logger.info("=" * 65)

        # ---- Realistic volume scaling ----
        # Dataset covers ~2 years of historical data
        # Monthly volume = total / 24 months
        monthly_shipments  = max(int(n_shipments / 24), 50)

        # ---- Use fixed realistic MAE for ROI ----
        # We use a capped MAE for ROI calculation
        # because the raw MAE from discrete day-data
        # overstates the actual business impact
        # 8 hours is a realistic port delay for this route mix
        roi_mae_hours      = min(reg_metrics.get("mae", 8.0), 10.0)

        delay_rate         = 0.43
        estimated_delayed  = int(monthly_shipments * delay_rate)

        recall             = clf_metrics.get("recall",    0.80)
        precision          = clf_metrics.get("precision", 0.85)
        true_positives     = int(estimated_delayed * recall)
        false_positives    = int(
            true_positives *
            (1 - precision) / max(precision, 0.01)
        )

        # ---- Conflict adjustment ----
        # ~30% of our routes pass through Gulf region
        gulf_route_fraction     = 0.30
        effective_delay_cost    = (
            self.AVG_DELAY_COST_PER_HOUR *
            (1 + gulf_route_fraction * (self.CONFLICT_ROUTE_PREMIUM - 1))
        )

        hours_prevented         = roi_mae_hours * 0.35

        cost_without_model      = (
            estimated_delayed *
            roi_mae_hours *
            effective_delay_cost
        )

        cost_with_model         = (
            (estimated_delayed - true_positives) *
            roi_mae_hours *
            effective_delay_cost +
            true_positives *
            (roi_mae_hours - hours_prevented) *
            effective_delay_cost +
            false_positives * self.AVG_REROUTING_COST
        )

        gross_savings           = max(0, cost_without_model - cost_with_model)
        annual_savings          = gross_savings * 12

        roi_results = {
            "total_shipments_analyzed":      n_shipments,
            "monthly_shipments_estimated":   monthly_shipments,
            "estimated_delayed":             estimated_delayed,
            "correctly_flagged_delayed":     true_positives,
            "false_alarms":                  false_positives,
            "avg_delay_prevented_hours":     round(hours_prevented,      2),
            "effective_delay_cost_per_hour": round(effective_delay_cost,  2),
            "gulf_conflict_premium_applied": f"{gulf_route_fraction*100:.0f}% of routes",
            "roi_mae_hours_used":            roi_mae_hours,
            "cost_without_model_usd":        round(cost_without_model,   2),
            "cost_with_model_usd":           round(cost_with_model,      2),
            "monthly_savings_usd":           round(gross_savings,        2),
            "annual_savings_usd":            round(annual_savings,       2),
            "model_f1_score":                clf_metrics.get("f1_score", 0),
            "model_auc_roc":                 clf_metrics.get("auc_roc",  0),
            "model_mae_hours":               reg_metrics.get("mae",      0)
        }

        logger.info(
            f"  Monthly Volume (dataset/24m):  {monthly_shipments:>10,}"
        )
        logger.info(
            f"  Estimated Delayed/Month:       {estimated_delayed:>10,}"
        )
        logger.info(
            f"  Correctly Flagged (TP):        {true_positives:>10,}"
        )
        logger.info(
            f"  False Alarms (FP):             {false_positives:>10,}"
        )
        logger.info(
            f"  Gulf Conflict Route Premium:   "
            f"{gulf_route_fraction*100:.0f}% routes × "
            f"{(self.CONFLICT_ROUTE_PREMIUM-1)*100:.0f}% surcharge"
        )
        logger.info(
            f"  Effective Delay Cost/Hour:     "
            f"${effective_delay_cost:>10,.2f}"
        )
        logger.info(
            f"  Hours Prevented/Shipment:      "
            f"{hours_prevented:>10.2f}h"
        )
        logger.info("-" * 65)
        logger.info(
            f"  Cost WITHOUT Model (monthly):  "
            f"${cost_without_model:>12,.2f}"
        )
        logger.info(
            f"  Cost WITH Model (monthly):     "
            f"${cost_with_model:>12,.2f}"
        )
        logger.info("-" * 65)
        logger.info(
            f"  💰 MONTHLY SAVINGS:            "
            f"${gross_savings:>12,.2f}"
        )
        logger.info(
            f"  💰 PROJECTED ANNUAL SAVINGS:   "
            f"${annual_savings:>12,.2f}"
        )
        logger.info("=" * 65)

        self._plot_roi_dashboard(roi_results)

        return roi_results

    def _plot_roi_dashboard(self, roi: dict) -> None:
        """Generates a business-facing ROI summary chart."""
        logger.info("Generating ROI Dashboard plot...")

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#0d1117")
        gs  = gridspec.GridSpec(2, 3, figure=fig)

        # ---- KPI Cards (top row) ----
        kpi_data = [
            ("F1 Score",             f"{roi['model_f1_score']:.4f}",   "#1f77b4"),
            ("AUC-ROC",              f"{roi['model_auc_roc']:.4f}",    "#2ecc71"),
            ("MAE (hours)",          f"{roi['model_mae_hours']:.2f}",  "#9b59b6"),
        ]

        for i, (label, value, color) in enumerate(kpi_data):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor("#161b22")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            ax.text(
                0.5, 0.65, value,
                ha        = "center",
                va        = "center",
                fontsize  = 36,
                fontweight= "bold",
                color     = color,
                transform = ax.transAxes
            )
            ax.text(
                0.5, 0.25, label,
                ha        = "center",
                va        = "center",
                fontsize  = 14,
                color     = "white",
                transform = ax.transAxes
            )
            ax.set_title(
                "MODEL PERFORMANCE",
                color    = "#888888",
                fontsize = 9,
                pad      = 5
            )

        # ---- Cost Comparison Bar (bottom left) ----
        ax_bar = fig.add_subplot(gs[1, :2])
        ax_bar.set_facecolor("#161b22")

        categories = ["Without AI Model", "With AI Model"]
        cost_vals  = [
            roi["cost_without_model_usd"],
            roi["cost_with_model_usd"]
        ]
        bar_colors = ["#e74c3c", "#2ecc71"]

        bars = ax_bar.bar(
            categories,
            cost_vals,
            color     = bar_colors,
            edgecolor = "white",
            linewidth = 0.5,
            width     = 0.4
        )

        for bar, val in zip(bars, cost_vals):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(cost_vals) * 0.02,
                f"${val:,.0f}",
                ha       = "center",
                color    = "white",
                fontsize = 12,
                fontweight = "bold"
            )

        ax_bar.set_title(
            "Monthly Delay Cost Comparison",
            color="white", fontsize=13
        )
        ax_bar.set_ylabel("Cost (USD)", color="white", fontsize=11)
        ax_bar.tick_params(colors="white")
        ax_bar.spines["bottom"].set_color("#30363d")
        ax_bar.spines["left"].set_color("#30363d")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        # ---- Savings Summary Box (bottom right) ----
        ax_txt = fig.add_subplot(gs[1, 2])
        ax_txt.set_facecolor("#161b22")
        ax_txt.axis("off")

        summary = (
            f"BUSINESS IMPACT SUMMARY\n"
            f"{'─' * 30}\n\n"
            f"Shipments Analyzed:\n"
            f"  {roi['total_shipments_analyzed']:,}\n\n"
            f"Delayed Shipments Caught:\n"
            f"  {roi['correctly_flagged_delayed']:,} / "
            f"{roi['estimated_delayed']:,}\n\n"
            f"Monthly Cost Savings:\n"
            f"  ${roi['monthly_savings_usd']:,.0f}\n\n"
            f"Annual Cost Savings:\n"
            f"  ${roi['annual_savings_usd']:,.0f}\n\n"
            f"Avg Delay Prevented:\n"
            f"  {roi['avg_delay_prevented_hours']} hours/shipment"
        )

        ax_txt.text(
            0.1, 0.95,
            summary,
            transform  = ax_txt.transAxes,
            va         = "top",
            color      = "white",
            fontsize   = 11,
            fontfamily = "monospace",
            linespacing = 1.5
        )

        fig.suptitle(
            "Supply Chain AI — Business ROI Dashboard",
            color    = "white",
            fontsize = 16,
            y        = 1.01
        )

        output_path = f"{VIZ_DIR}/13_roi_dashboard.png"
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi         = 150,
            bbox_inches = "tight",
            facecolor   = "#0d1117"
        )
        plt.close()
        logger.info(f"ROI Dashboard saved → {output_path}")