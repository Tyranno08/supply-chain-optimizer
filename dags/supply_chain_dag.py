# dags/supply_chain_dag.py
# ============================================
# APACHE AIRFLOW DAG
# Supply Chain Daily Ingestion & Retraining
# ============================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

# ============================================
# DEFAULT DAG ARGUMENTS
# ============================================
default_args = {
    "owner":            "supply_chain_ai",
    "depends_on_past":  False,
    "start_date":       days_ago(1),
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ============================================
# PYTHON CALLABLES (task functions)
# ============================================

def task_run_data_ingestion(**context):
    """Step 1: Pull fresh weather + news data from APIs."""
    import subprocess, sys
    logger.info("Starting data ingestion pipeline...")
    result = subprocess.run(
        [sys.executable, "src/ingestion/data_ingestion.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ingestion failed:\n{result.stderr}")
    logger.info("Data ingestion complete.")
    logger.info(result.stdout)


def task_run_feature_engineering(**context):
    """Step 2: Rebuild feature matrix from new data."""
    import subprocess, sys
    logger.info("Running feature engineering pipeline...")
    result = subprocess.run(
        [sys.executable, "src/features/feature_engineering.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Feature engineering failed:\n{result.stderr}")
    logger.info("Feature engineering complete.")


def task_run_drift_detection(**context):
    """Step 3: Detect data drift in new feature matrix."""
    import subprocess, sys
    logger.info("Running drift detection...")
    result = subprocess.run(
        [sys.executable, "src/monitoring/drift_detector.py"],
        capture_output=True, text=True
    )
    logger.info(result.stdout)

    # Parse drift result and push to XCom for branching
    drift_detected = "DRIFT DETECTED" in result.stdout
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    logger.info(f"Drift detected: {drift_detected}")
    return drift_detected


def task_branch_on_drift(**context):
    """Step 4: Branch — retrain if drift, skip if stable."""
    drift_detected = context["ti"].xcom_pull(
        task_ids="drift_detection", key="drift_detected"
    )
    if drift_detected:
        logger.info("Drift detected → triggering retraining.")
        return "retrain_models"
    else:
        logger.info("No drift → skipping retraining.")
        return "skip_retraining"


def task_retrain_models(**context):
    """Step 5a: Retrain all models with MLflow tracking."""
    import subprocess, sys
    logger.info("Retraining all models...")
    result = subprocess.run(
        [sys.executable, "src/models/run_model_pipeline.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Retraining failed:\n{result.stderr}")
    logger.info("Retraining complete.")
    logger.info(result.stdout)


def task_promote_best_model(**context):
    """Step 6: Promote best MLflow model to production."""
    import subprocess, sys
    logger.info("Promoting best model to production...")
    result = subprocess.run(
        [sys.executable, "src/monitoring/model_promoter.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Promotion failed:\n{result.stderr}")
    logger.info(result.stdout)


def task_run_evaluation(**context):
    """Step 7: Re-run evaluation and refresh dashboard reports."""
    import subprocess, sys
    logger.info("Running evaluation pipeline...")
    result = subprocess.run(
        [sys.executable, "src/models/run_evaluation.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.warning(f"Evaluation warning:\n{result.stderr}")
    logger.info("Evaluation complete.")


def task_health_check(**context):
    """Step 8: Verify API is still healthy after retraining."""
    import requests
    try:
        resp = requests.get("http://localhost:8000/health", timeout=10)
        if resp.status_code == 200:
            logger.info("API health check PASSED.")
        else:
            raise RuntimeError(f"API health check failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"API health check could not reach server: {e}")


# ============================================
# DAG DEFINITION
# ============================================
with DAG(
    dag_id              = "supply_chain_daily_pipeline",
    default_args        = default_args,
    description         = "Daily supply chain data ingestion, drift detection, and conditional retraining",
    schedule_interval   = "0 2 * * *",   # Run at 2:00 AM UTC daily
    catchup             = False,
    max_active_runs     = 1,
    tags                = ["supply_chain", "mlops", "production"],
) as dag:

    # ---- START ----
    start = DummyOperator(task_id="start")

    # ---- Step 1: Ingest new data ----
    ingest = PythonOperator(
        task_id         = "data_ingestion",
        python_callable = task_run_data_ingestion,
        provide_context = True,
    )

    # ---- Step 2: Feature engineering ----
    feature_eng = PythonOperator(
        task_id         = "feature_engineering",
        python_callable = task_run_feature_engineering,
        provide_context = True,
    )

    # ---- Step 3: Drift detection ----
    drift_detect = PythonOperator(
        task_id         = "drift_detection",
        python_callable = task_run_drift_detection,
        provide_context = True,
    )

    # ---- Step 4: Branch ----
    branch = BranchPythonOperator(
        task_id         = "branch_on_drift",
        python_callable = task_branch_on_drift,
        provide_context = True,
    )

    # ---- Step 5a: Retrain (if drift) ----
    retrain = PythonOperator(
        task_id         = "retrain_models",
        python_callable = task_retrain_models,
        provide_context = True,
    )

    # ---- Step 5b: Skip retraining ----
    skip_retrain = DummyOperator(task_id="skip_retraining")

    # ---- Step 6: Promote best model ----
    promote = PythonOperator(
        task_id         = "promote_best_model",
        python_callable = task_promote_best_model,
        provide_context = True,
        trigger_rule    = "none_failed_min_one_success",
    )

    # ---- Step 7: Re-evaluate ----
    evaluate = PythonOperator(
        task_id         = "run_evaluation",
        python_callable = task_run_evaluation,
        provide_context = True,
        trigger_rule    = "none_failed_min_one_success",
    )

    # ---- Step 8: Health check ----
    health = PythonOperator(
        task_id         = "api_health_check",
        python_callable = task_health_check,
        provide_context = True,
    )

    # ---- END ----
    end = DummyOperator(
        task_id      = "end",
        trigger_rule = "none_failed_min_one_success",
    )

    # ============================================
    # DAG DEPENDENCY GRAPH
    # ============================================
    start >> ingest >> feature_eng >> drift_detect >> branch
    branch >> retrain    >> promote >> evaluate >> health >> end
    branch >> skip_retrain           >> evaluate >> health >> end
