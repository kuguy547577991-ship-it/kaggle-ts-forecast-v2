"""Kaggle Time Series Forecast Pipeline — hand-written equivalent of Kedro's
pipeline_registry.py with __default__ = data_processing → feature_engineering
→ model_training → model_evaluation.

Usage:
    py pipelines/ts_forecast/pipeline.py
    py pipelines/ts_forecast/pipeline.py --config local
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.ts_forecast.nodes.preprocessing import (
    check_missing_values,
    encode_categorical_columns,
    fill_missing_values,
    load_raw_data,
)
from pipelines.ts_forecast.nodes.feature_engineering import (
    create_datetime_features,
    create_diff_features,
    create_ewm_features,
    create_lag_features,
    create_rolling_features,
    train_test_split_by_time,
)
from pipelines.ts_forecast.nodes.training import (
    prepare_training_data,
    save_models,
    train_lightgbm,
    train_random_forest,
    train_ridge,
    train_xgboost,
)
from pipelines.ts_forecast.nodes.evaluation import (
    create_feature_importance_df,
    evaluate_all_models,
    plot_predictions,
    save_evaluation_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ts_forecast")


def load_config(env: str = "base") -> dict:
    """Load YAML config with env override.

    Equivalent to Kedro's OmegaConfigLoader merging conf/base + conf/local.
    """
    config_dir = Path(__file__).parent / "config"
    cfg = {}

    base_path = config_dir / "base.yaml"
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f)

    env_path = config_dir / f"{env}.yaml"
    if env != "base" and env_path.exists():
        with open(env_path) as f:
            env_cfg = yaml.safe_load(f)
            cfg = _deep_merge(cfg, env_cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def run_ts_forecast_pipeline(env: str = "base") -> None:
    """Execute the full time series forecast pipeline."""
    cfg = load_config(env)
    logger.info(f"Starting TS forecast pipeline (env={env})")

    mlflow.set_experiment("kaggle_ts_forecast")

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run: {run.info.run_id}")

        # ═══════════════════════════════════════════════════
        # STEP 1: Data Processing
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("STEP 1: Data Processing")
        logger.info("=" * 50)

        data_cfg = cfg["data"]
        prep_cfg = cfg["preprocessing"]

        df = load_raw_data(data_cfg["raw_path"], date_column=data_cfg["date_column"])
        df = check_missing_values(df)
        df = fill_missing_values(df, method=prep_cfg["fill_method"])
        df = encode_categorical_columns(df, columns=prep_cfg["categorical_columns"])
        logger.info(f"Data processing complete: {df.shape}")

        # ═══════════════════════════════════════════════════
        # STEP 2: Feature Engineering
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 50)

        feat_cfg = cfg["features"]
        target_cfg = cfg["target"]
        target = target_cfg["name"]

        if feat_cfg.get("datetime_features", True):
            df = create_datetime_features(df, date_column=data_cfg["date_column"])

        df = create_lag_features(
            df, target_column=target,
            lag_periods=feat_cfg["lag_periods"],
            groupby_columns=target_cfg["groupby_columns"],
        )
        df = create_rolling_features(
            df, target_column=target,
            windows=feat_cfg["rolling_windows"],
            groupby_columns=target_cfg["groupby_columns"],
        )
        df = create_diff_features(df, target_column=target,
                                  periods=feat_cfg["diff_periods"])
        df = create_ewm_features(df, target_column=target,
                                 spans=feat_cfg["ewm_spans"])

        train_data, test_data = train_test_split_by_time(
            df, date_column=data_cfg["date_column"],
            test_ratio=cfg["split"]["test_ratio"],
        )
        logger.info(f"Feature engineering complete: {len(df.columns)} total features")

        mlflow.log_params({
            "n_features": len(df.columns),
            "n_train": len(train_data),
            "n_test": len(test_data),
            "test_ratio": cfg["split"]["test_ratio"],
        })

        # ═══════════════════════════════════════════════════
        # STEP 3: Model Training
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("STEP 3: Model Training")
        logger.info("=" * 50)

        exclude = feat_cfg["exclude_columns"]
        X_train, y_train, feature_cols = prepare_training_data(
            train_data, target_column=target, exclude_columns=exclude,
        )

        train_cfg = cfg["training"]
        model_cfg = cfg["models"]

        lgb_model = train_lightgbm(
            X_train, y_train, model_cfg["lightgbm"],
            cv_splits=train_cfg["cv_splits"],
            early_stopping_rounds=train_cfg["early_stopping_rounds"],
        )
        xgb_model = train_xgboost(
            X_train, y_train, model_cfg["xgboost"],
            cv_splits=train_cfg["cv_splits"],
        )
        rf_model = train_random_forest(X_train, y_train, model_cfg["random_forest"])
        ridge_model = train_ridge(X_train, y_train, model_cfg["ridge"])

        save_models(lgb_model, xgb_model, rf_model, ridge_model)

        # ═══════════════════════════════════════════════════
        # STEP 4: Evaluation
        # ═══════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("STEP 4: Evaluation")
        logger.info("=" * 50)

        eval_cfg = cfg["evaluation"]
        X_test, y_test, _ = prepare_training_data(
            test_data, target_column=target, exclude_columns=exclude,
        )

        results = evaluate_all_models(
            lgb_model, xgb_model, rf_model, ridge_model,
            X_test, y_test,
        )

        # Log all metrics to MLflow
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)

        save_evaluation_results(results, eval_cfg["metrics_output"])

        # Feature importance
        feat_imp = create_feature_importance_df(
            lgb_model, feature_cols, top_n=eval_cfg["top_n_features"],
        )
        logger.info(f"Top 5 features:\n{feat_imp.head()}")

        # Plot predictions
        fig = plot_predictions(
            test_data, lgb_model, xgb_model,
            target_column=target,
            date_column=data_cfg["date_column"],
            exclude_columns=exclude,
        )
        fig.write_html("data/08_reporting/predictions.html")
        mlflow.log_artifact("data/08_reporting/metrics.json")
        mlflow.log_artifact("data/08_reporting/predictions.html")

        # Summary
        best = min(results.items(), key=lambda x: x[1]["rmse"])
        logger.info(f"Pipeline complete. Best model: {best[0]} (RMSE: {best[1]['rmse']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kaggle TS Forecast pipeline")
    parser.add_argument("--config", "-c", default="base",
                        help="Config environment (base, local)")
    args = parser.parse_args()
    run_ts_forecast_pipeline(env=args.config)
