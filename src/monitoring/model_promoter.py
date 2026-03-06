# src/monitoring/model_promoter.py
# ============================================
# MODEL VERSIONING & AUTO-PROMOTION
# Compares new MLflow model runs against the
# current production model. Promotes the best
# model to "Production" stage automatically.
# ============================================

import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import get_logger

logger = get_logger("ModelPromoter")

# ---- Promotion thresholds ----
MIN_AUC_ROC_IMPROVEMENT = 0.005   # Must improve AUC-ROC by at least 0.5%
MIN_F1_IMPROVEMENT      = 0.005   # Must improve F1 by at least 0.5%
PROMOTION_REPORT_PATH   = "models/promotion_report.json"


# ============================================
# FETCH MLFLOW RUNS
# ============================================
def get_all_runs(experiment_name: str = "supply_chain_delay_prediction") -> list:
    """Fetches all MLflow runs for the classifier, sorted by AUC-ROC."""
    import mlflow

    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found.")
            return []

        runs = client.search_runs(
            experiment_ids = [experiment.experiment_id],
            filter_string  = "tags.mlflow.runName = 'xgboost_classifier'",
            order_by       = ["metrics.auc_roc DESC"],
            max_results    = 20
        )
        return runs

    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return []


# ============================================
# GET CURRENT PRODUCTION MODEL METRICS
# ============================================
def get_production_metrics(model_name: str = "supply_chain_delay_classifier") -> dict:
    """Fetches metrics of the currently Production-staged model."""
    import mlflow

    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            logger.info("No Production model found. Any new model qualifies for promotion.")
            return {}

        prod_version = prod_versions[0]
        run          = client.get_run(prod_version.run_id)
        metrics      = run.data.metrics
        logger.info(f"Current Production model: v{prod_version.version}")
        logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
        logger.info(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
        return metrics

    except Exception as e:
        logger.warning(f"Could not fetch production metrics: {e}")
        return {}


# ============================================
# PROMOTE MODEL
# ============================================
def promote_model(
    model_name:  str,
    run_id:      str,
    new_version: str,
    reason:      str
) -> bool:
    """Transitions a model version to Production and archives the old one."""
    import mlflow

    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    try:
        # Archive existing production versions
        current_prod = client.get_latest_versions(model_name, stages=["Production"])
        for v in current_prod:
            client.transition_model_version_stage(
                name    = model_name,
                version = v.version,
                stage   = "Archived"
            )
            logger.info(f"Archived previous production model v{v.version}")

        # Promote new version
        client.transition_model_version_stage(
            name    = model_name,
            version = new_version,
            stage   = "Production"
        )

        # Add description tag
        client.update_model_version(
            name        = model_name,
            version     = new_version,
            description = f"Auto-promoted on {datetime.now().isoformat()}. Reason: {reason}"
        )

        logger.info(f"Model '{model_name}' v{new_version} promoted to Production!")
        return True

    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        return False


# ============================================
# MAIN PROMOTION LOGIC
# ============================================
def run_model_promotion() -> dict:
    """
    Main function:
    1. Fetches all classifier runs
    2. Compares best new run vs current production
    3. Promotes if performance improved beyond threshold
    """
    import mlflow

    logger.info("=" * 55)
    logger.info("  MODEL PROMOTION PIPELINE — STARTING")
    logger.info("=" * 55)

    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    model_name = "supply_chain_delay_classifier"

    # ---- Get all runs ----
    runs = get_all_runs()
    if not runs:
        logger.warning("No runs found. Skipping promotion.")
        return {"promoted": False, "reason": "No runs found"}

    best_run     = runs[0]
    best_metrics = best_run.data.metrics
    best_run_id  = best_run.info.run_id

    logger.info(f"Best candidate run: {best_run_id}")
    logger.info(f"  AUC-ROC: {best_metrics.get('auc_roc', 'N/A')}")
    logger.info(f"  F1 Score: {best_metrics.get('f1_score', 'N/A')}")

    # ---- Get current production metrics ----
    prod_metrics = get_production_metrics(model_name)

    # ---- Compare and decide ----
    new_auc = best_metrics.get("auc_roc", 0)
    new_f1  = best_metrics.get("f1_score", 0)
    old_auc = prod_metrics.get("auc_roc", 0)
    old_f1  = prod_metrics.get("f1_score", 0)

    auc_improvement = new_auc - old_auc
    f1_improvement  = new_f1 - old_f1

    should_promote = (
        not prod_metrics  # No production model exists yet
        or auc_improvement >= MIN_AUC_ROC_IMPROVEMENT
        or f1_improvement  >= MIN_F1_IMPROVEMENT
    )

    promotion_report = {
        "timestamp":          datetime.now().isoformat(),
        "promoted":           False,
        "model_name":         model_name,
        "candidate_run_id":   best_run_id,
        "candidate_auc_roc":  round(new_auc, 4),
        "candidate_f1":       round(new_f1, 4),
        "production_auc_roc": round(old_auc, 4),
        "production_f1":      round(old_f1, 4),
        "auc_improvement":    round(auc_improvement, 4),
        "f1_improvement":     round(f1_improvement, 4),
        "thresholds": {
            "min_auc_improvement": MIN_AUC_ROC_IMPROVEMENT,
            "min_f1_improvement":  MIN_F1_IMPROVEMENT
        }
    }

    if should_promote:
        reason = (
            "No production model existed" if not prod_metrics
            else f"AUC improved by {auc_improvement:.4f}, F1 improved by {f1_improvement:.4f}"
        )

        # Find the model version for this run
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            run_version = next(
                (v.version for v in versions if v.run_id == best_run_id), None
            )

            if run_version:
                success = promote_model(model_name, best_run_id, run_version, reason)
                promotion_report["promoted"]        = success
                promotion_report["promoted_version"] = run_version
                promotion_report["reason"]           = reason
            else:
                logger.warning("Could not find model version for best run.")
                promotion_report["reason"] = "Version not found in registry"

        except Exception as e:
            logger.error(f"Error during promotion: {e}")
            promotion_report["reason"] = str(e)

    else:
        reason = (
            f"New model did not improve sufficiently. "
            f"AUC Δ={auc_improvement:.4f} (need >{MIN_AUC_ROC_IMPROVEMENT}), "
            f"F1 Δ={f1_improvement:.4f} (need >{MIN_F1_IMPROVEMENT})"
        )
        promotion_report["reason"] = reason
        logger.info(f"No promotion needed: {reason}")

    # ---- Save report ----
    os.makedirs("models", exist_ok=True)
    with open(PROMOTION_REPORT_PATH, "w") as f:
        json.dump(promotion_report, f, indent=2)

    logger.info("=" * 55)
    logger.info("  PROMOTION SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Promoted:    {promotion_report['promoted']}")
    logger.info(f"  Reason:      {promotion_report.get('reason', 'N/A')}")
    logger.info(f"  Report saved → {PROMOTION_REPORT_PATH}")
    logger.info("=" * 55)

    return promotion_report


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    report = run_model_promotion()
    print(json.dumps(report, indent=2))
