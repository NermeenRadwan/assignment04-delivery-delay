"""
Airflow Data Pipeline DAG: data_pipeline
Orchestrates daily Medallion Architecture:
extract_to_bronze -> transform_to_silver -> prepare_gold -> validate_gold
Date-aware using logical execution date (ds), idempotent, and configured with retries=2.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root is in sys.path for Airflow workers
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.etl import (
    extract_to_bronze,
    prepare_gold,
    transform_to_silver,
    validate_gold,
)


def _extract_task(ds: str, **kwargs) -> str:
    """Extract raw daily CSV to Bronze with optional transient retry simulation."""
    ti = kwargs.get("task_instance")
    simulate_date = os.environ.get("SIMULATE_RETRY_ON_DATE")
    # Simulate a transient failure on attempt 1 if requested
    if simulate_date and simulate_date == ds and ti and ti.try_number == 1:
        raise RuntimeError(
            f"[Simulated Retry Test] Transient network/storage glitch on attempt 1 for {ds}. "
            "Airflow will automatically retry this task!"
        )
    return extract_to_bronze(date_str=ds)


def _silver_task(ds: str, **kwargs) -> str:
    """Transform Bronze data to Silver."""
    return transform_to_silver(date_str=ds)


def _gold_task(ds: str, **kwargs) -> str:
    """Prepare Gold data with engineered features."""
    return prepare_gold(date_str=ds)


def _validate_task(ds: str, **kwargs) -> bool:
    """Validate Gold data against quality gates."""
    return validate_gold(date_str=ds)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    description="Daily Medallion ETL pipeline for delivery delay prediction",
    schedule_interval="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["data_engineering", "medallion", "assignment04"],
) as dag:

    extract_to_bronze_task = PythonOperator(
        task_id="extract_to_bronze",
        python_callable=_extract_task,
        op_kwargs={"ds": "{{ ds }}"},
        provide_context=True,
    )

    transform_to_silver_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=_silver_task,
        op_kwargs={"ds": "{{ ds }}"},
        provide_context=True,
    )

    prepare_gold_task = PythonOperator(
        task_id="prepare_gold",
        python_callable=_gold_task,
        op_kwargs={"ds": "{{ ds }}"},
        provide_context=True,
    )

    validate_gold_task = PythonOperator(
        task_id="validate_gold",
        python_callable=_validate_task,
        op_kwargs={"ds": "{{ ds }}"},
        provide_context=True,
    )

    # Task dependencies:
    # extract_to_bronze -> transform_to_silver -> prepare_gold -> validate_gold
    (
        extract_to_bronze_task
        >> transform_to_silver_task
        >> prepare_gold_task
        >> validate_gold_task
    )
