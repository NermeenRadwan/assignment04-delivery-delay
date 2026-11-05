"""
Airflow ML Pipeline DAG: ml_training_pipeline
Orchestrates model lifecycle:
load_gold_data -> train_model -> evaluate_model -> save_model
Keeps core ML logic in src/train.py and orchestrates via Airflow.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import (
    step_evaluate_model,
    step_load_gold_data,
    step_save_model,
    step_train_model,
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=15),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="Machine Learning training and evaluation pipeline for delivery delay prediction",
    schedule_interval=None,  # Triggered manually once Gold data is validated
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["mlops", "training", "assignment04"],
) as dag:

    load_gold_data_task = PythonOperator(
        task_id="load_gold_data",
        python_callable=step_load_gold_data,
        op_kwargs={"gold_dir": "data/gold", "ml_dir": "data/ml"},
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=step_train_model,
        op_kwargs={"ml_dir": "data/ml"},
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=step_evaluate_model,
        op_kwargs={"ml_dir": "data/ml", "metrics_path": "metrics/metrics.json"},
    )

    save_model_task = PythonOperator(
        task_id="save_model",
        python_callable=step_save_model,
        op_kwargs={
            "ml_dir": "data/ml",
            "final_model_path": "models/delivery_delay_model.joblib",
        },
    )

    # Task dependencies:
    # load_gold_data -> train_model -> evaluate_model -> save_model
    (load_gold_data_task >> train_model_task >> evaluate_model_task >> save_model_task)
