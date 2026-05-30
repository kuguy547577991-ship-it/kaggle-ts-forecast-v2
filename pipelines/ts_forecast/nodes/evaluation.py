"""Model evaluation nodes — ported from kaggle-ts-forecast model_evaluation/nodes.py."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def prepare_evaluation_data(
    test_data: pd.DataFrame,
    target_column: str,
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare test data for evaluation."""
    exclude = exclude_columns or []
    exclude.append(target_column)

    feature_cols = [
        c for c in test_data.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    df = test_data.dropna(subset=feature_cols + [target_column])
    X = df[feature_cols].values
    y = df[target_column].values

    return X, y, feature_cols


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "r2": r2_score(y_test, y_pred),
    }

    logger.info(
        f"{model_name}: MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%"
    )
    return {model_name: metrics}


def evaluate_all_models(
    lgb_model: Any,
    xgb_model: Any,
    rf_model: Any,
    ridge_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate all models and return comparison."""
    results = {}
    for name, model in [
        ("lightgbm", lgb_model),
        ("xgboost", xgb_model),
        ("random_forest", rf_model),
        ("ridge", ridge_model),
    ]:
        results.update(evaluate_model(model, X_test, y_test, name))

    best = min(results.items(), key=lambda x: x[1]["rmse"])
    logger.info(f"Best model: {best[0]} (RMSE: {best[1]['rmse']:.4f})")
    return results


def plot_predictions(
    test_data: pd.DataFrame,
    lgb_model: Any,
    xgb_model: Any,
    target_column: str,
    date_column: str,
    exclude_columns: list[str] | None = None,
) -> go.Figure:
    """Create an interactive plot of actual vs predicted values."""
    exclude = exclude_columns or []
    exclude.append(target_column)

    feature_cols = [
        c for c in test_data.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    df = test_data.dropna(subset=feature_cols + [target_column])
    X = df[feature_cols].values
    y_true = df[target_column].values
    dates = df[date_column].values

    y_lgb = lgb_model.predict(X)
    y_xgb = xgb_model.predict(X)

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, mode="lines",
        name="Actual", line=dict(color="black", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_lgb, mode="lines",
        name="LightGBM", line=dict(dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_xgb, mode="lines",
        name="XGBoost", line=dict(dash="dot"),
    ))
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title=target_column,
        template="plotly_white",
    )
    return fig


def save_evaluation_results(
    results: dict[str, dict[str, float]],
    output_path: str = "data/08_reporting/metrics.json",
) -> str:
    """Save evaluation metrics to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved evaluation results to {output_path}")
    return output_path


def create_feature_importance_df(
    lgb_model: Any,
    feature_columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract top feature importances from LightGBM model."""
    importances = lgb_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    return feat_imp
