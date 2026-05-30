"""Time series specific metrics — adapted from kaggle-ts-forecast utils/metrics.py."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Compute MASE (Mean Absolute Scaled Error)."""
    naive_errors = np.mean(np.abs(np.diff(y_train, n=seasonality)))
    mae = mean_absolute_error(y_true, y_pred)
    return float(mae / (naive_errors + 1e-8))


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute SMAPE (Symmetric MAPE) in percentage."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Compute a comprehensive set of regression/time series metrics."""
    metrics = {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}mse": mean_squared_error(y_true, y_pred),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        f"{prefix}smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
    }
    if y_train is not None:
        metrics[f"{prefix}mase"] = mean_absolute_scaled_error(y_true, y_pred, y_train)
    return metrics
