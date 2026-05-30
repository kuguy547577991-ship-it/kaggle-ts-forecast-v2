"""Model training nodes — ported from kaggle-ts-forecast model_training/nodes.py."""

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def prepare_training_data(
    train_data: pd.DataFrame,
    target_column: str,
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Separate features and target for training."""
    exclude = exclude_columns or []
    exclude.append(target_column)

    feature_cols = [
        c for c in train_data.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    df = train_data.dropna(subset=feature_cols + [target_column])
    X = df[feature_cols].values
    y = df[target_column].values

    logger.info(f"Training data: X={X.shape}, y={y.shape}")
    return X, y, feature_cols


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
    cv_splits: int = 5,
    early_stopping_rounds: int = 50,
) -> Any:
    """Train a LightGBM regression model with time series CV."""
    default_params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
        )
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        scores.append(score)
        logger.info(f"LightGBM Fold {fold + 1}: MAE = {score:.4f}")

    cv_mean = np.mean(scores)
    logger.info(f"LightGBM CV MAE: {cv_mean:.4f} (+/- {np.std(scores):.4f})")
    mlflow.log_metric("lgb_cv_mae_mean", float(cv_mean))
    mlflow.log_metric("lgb_cv_mae_std", float(np.std(scores)))
    return model


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
    cv_splits: int = 5,
) -> Any:
    """Train an XGBoost regression model."""
    default_params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        scores.append(score)
        logger.info(f"XGBoost Fold {fold + 1}: MAE = {score:.4f}")

    cv_mean = np.mean(scores)
    logger.info(f"XGBoost CV MAE: {cv_mean:.4f} (+/- {np.std(scores):.4f})")
    mlflow.log_metric("xgb_cv_mae_mean", float(cv_mean))
    mlflow.log_metric("xgb_cv_mae_std", float(np.std(scores)))
    return model


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a Random Forest regression model."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestRegressor(**default_params)
    model.fit(X, y)
    return model


def train_ridge(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a Ridge regression model (linear baseline)."""
    default_params = {"alpha": 1.0, "random_state": 42}
    if params:
        default_params.update(params)

    model = Ridge(**default_params)
    model.fit(X, y)
    return model


def save_models(
    lgb_model: Any,
    xgb_model: Any,
    rf_model: Any,
    ridge_model: Any,
    model_dir: str = "data/06_models",
) -> dict[str, str]:
    """Save trained models to disk."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    models = {
        "lightgbm": lgb_model,
        "xgboost": xgb_model,
        "random_forest": rf_model,
        "ridge": ridge_model,
    }
    paths = {}
    for name, model in models.items():
        path = f"{model_dir}/{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        paths[name] = path
        logger.info(f"Saved {name} model to {path}")

    return paths
