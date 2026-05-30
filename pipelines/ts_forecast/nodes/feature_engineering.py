"""Feature engineering nodes — ported from kaggle-ts-forecast feature_engineering/nodes.py."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_datetime_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Extract datetime features from a date column."""
    df = df.copy()
    dates = pd.to_datetime(df[date_column])
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["dayofweek"] = dates.dt.dayofweek
    df["quarter"] = dates.dt.quarter
    df["dayofyear"] = dates.dt.dayofyear
    df["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lag_periods: list[int] | None = None,
    groupby_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create lag features for the target column."""
    if lag_periods is None:
        lag_periods = [1, 2, 3, 7, 14, 28]

    df = df.copy()
    if groupby_columns:
        for lag in lag_periods:
            df[f"{target_column}_lag_{lag}"] = (
                df.groupby(groupby_columns)[target_column].shift(lag)
            )
    else:
        for lag in lag_periods:
            df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: list[int] | None = None,
    groupby_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create rolling window statistics for the target column."""
    if windows is None:
        windows = [7, 14, 30]

    df = df.copy()
    for window in windows:
        if groupby_columns:
            grp = df.groupby(groupby_columns)[target_column]
            df[f"{target_column}_rolling_mean_{window}"] = grp.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{target_column}_rolling_std_{window}"] = grp.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f"{target_column}_rolling_min_{window}"] = grp.transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f"{target_column}_rolling_max_{window}"] = grp.transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        else:
            rolled = df[target_column].rolling(window=window, min_periods=1)
            df[f"{target_column}_rolling_mean_{window}"] = rolled.mean()
            df[f"{target_column}_rolling_std_{window}"] = rolled.std()
            df[f"{target_column}_rolling_min_{window}"] = rolled.min()
            df[f"{target_column}_rolling_max_{window}"] = rolled.max()
    return df


def create_diff_features(
    df: pd.DataFrame,
    target_column: str,
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """Create differencing features for stationarity."""
    if periods is None:
        periods = [1, 7, 30]
    df = df.copy()
    for period in periods:
        df[f"{target_column}_diff_{period}"] = df[target_column].diff(period)
    return df


def create_ewm_features(
    df: pd.DataFrame,
    target_column: str,
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """Create exponentially weighted moving average features."""
    if spans is None:
        spans = [7, 14, 30]
    df = df.copy()
    for span in spans:
        df[f"{target_column}_ewm_{span}"] = (
            df[target_column].ewm(span=span, adjust=False).mean()
        )
    return df


def train_test_split_by_time(
    df: pd.DataFrame,
    date_column: str = "date",
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data chronologically."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test
