# src/models/run_model_pipeline.py
# ============================================
# MASTER MODEL TRAINING PIPELINE RUNNER
# Runs all Phase 5 steps in correct order
# ============================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.models.train_xgboost  import run_xgboost_training
from src.models.train_gnn      import run_gnn_training
from src.models.hybrid_ensemble import run_ensemble_demo
from src.utils import get_logger

logger = get_logger("ModelPipeline")


def run_all() -> None:
    logger.info("=" * 65)
    logger.info("  PHASE 5 — FULL MODEL TRAINING PIPELINE")
    logger.info("=" * 65)

    # Step 1: XGBoost (fast — runs in ~2 minutes)
    logger.info("STEP 1/3: Training XGBoost Models...")
    run_xgboost_training()

    # Step 2: GNN (medium — runs in ~5 minutes on CPU)
    logger.info("STEP 2/3: Training Graph Neural Network...")
    run_gnn_training()

    # Step 3: Hybrid ensemble demo + save config
    logger.info("STEP 3/3: Running Hybrid Ensemble Demo...")
    run_ensemble_demo()

    logger.info("=" * 65)
    logger.info("  PHASE 5 COMPLETE — ALL MODELS TRAINED & SAVED")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_all()