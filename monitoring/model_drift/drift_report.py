"""Model drift monitoring for Kaggle TS forecast."""

import logging
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, RegressionPerformanceMetrics
from evidently.report import Report

logger = logging.getLogger(__name__)


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = "sales",
    prediction_column: str = "sales_predicted",
    output_path: str = "data/monitoring/drift_report.html",
) -> None:
    column_mapping = ColumnMapping(
        target=target_column,
        prediction=prediction_column,
        numerical_features=reference_data.select_dtypes(
            include=["number"]
        ).columns.tolist(),
    )

    report = Report(metrics=[DataDriftTable(), RegressionPerformanceMetrics()])
    report.run(reference_data=reference_data, current_data=current_data,
               column_mapping=column_mapping)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)

    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["dataset_drift"]
    logger.info(f"Data drift share: {drift_share:.4f}")
