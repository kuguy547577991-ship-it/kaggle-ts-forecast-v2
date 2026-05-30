"""Airflow DAG for scheduled Kaggle TS forecast training."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="ts_forecast_training",
    default_args=default_args,
    schedule_interval="0 3 * * 1",  # Every Monday at 3 AM
    catchup=False,
    tags=["kaggle", "forecast"],
):

    train = BashOperator(
        task_id="train_model",
        bash_command="py pipelines/ts_forecast/pipeline.py --config local",
    )
