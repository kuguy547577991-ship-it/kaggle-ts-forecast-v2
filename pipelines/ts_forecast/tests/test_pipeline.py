"""Smoke tests for the TS forecast pipeline nodes."""

import numpy as np
import pandas as pd
import pytest

from pipelines.ts_forecast.nodes.preprocessing import (
    check_missing_values,
    fill_missing_values,
    load_raw_data,
)
from pipelines.ts_forecast.nodes.feature_engineering import (
    create_datetime_features,
    create_diff_features,
    create_lag_features,
    create_rolling_features,
    train_test_split_by_time,
)
from pipelines.ts_forecast.nodes.evaluation import evaluate_model


class TestPreprocessing:
    def test_fill_missing_values_ffill(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
        })
        result = fill_missing_values(df, method="ffill")
        assert result["a"].isnull().sum() == 0
        assert result.loc[1, "a"] == 1.0


class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        return pd.DataFrame({
            "date": dates,
            "sales": np.arange(50, dtype=float) + np.random.normal(0, 1, 50),
            "store": ["A"] * 50,
            "item": ["X"] * 50,
        })

    def test_create_datetime_features(self, sample_df):
        result = create_datetime_features(sample_df, date_column="date")
        assert "year" in result.columns
        assert "month" in result.columns
        assert "dayofweek" in result.columns

    def test_create_lag_features(self, sample_df):
        result = create_lag_features(sample_df, target_column="sales",
                                     lag_periods=[1, 2, 3],
                                     groupby_columns=["store", "item"])
        assert "sales_lag_1" in result.columns
        assert "sales_lag_3" in result.columns

    def test_create_rolling_features(self, sample_df):
        result = create_rolling_features(sample_df, target_column="sales",
                                         windows=[7],
                                         groupby_columns=["store", "item"])
        assert "sales_rolling_mean_7" in result.columns
        assert "sales_rolling_std_7" in result.columns

    def test_train_test_split(self, sample_df):
        train, test = train_test_split_by_time(sample_df, date_column="date",
                                                test_ratio=0.2)
        assert len(train) == 40
        assert len(test) == 10


class TestEvaluation:
    def test_evaluate_model(self):
        from sklearn.linear_model import LinearRegression
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression().fit(X, y)
        result = evaluate_model(model, X, y, "test_model")
        assert "test_model" in result
        assert result["test_model"]["r2"] > 0.99
