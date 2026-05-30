"""Preprocessing nodes — ported from kaggle-ts-forecast data_processing/nodes.py."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str, date_column: str = "date") -> pd.DataFrame:
    """Load raw CSV data and parse date column."""
    df = pd.read_csv(file_path)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Log missing value statistics."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_info = missing_info[missing_info["missing_count"] > 0]
    if not missing_info.empty:
        logger.warning(f"Missing values found:\n{missing_info}")
    else:
        logger.info("No missing values detected.")
    return df


def fill_missing_values(
    df: pd.DataFrame,
    method: str = "ffill",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Fill missing values using forward fill, interpolation, or median."""
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if col not in df.columns:
            continue
        if method == "ffill":
            df[col] = df[col].ffill().bfill()
        elif method == "interpolate":
            df[col] = df[col].interpolate(method="linear").ffill().bfill()
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_categorical_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Label-encode categorical columns."""
    cols = columns if columns else df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    return df


def scale_numeric_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    scaler_type: str = "standard",
) -> pd.DataFrame:
    """Scale numeric features using StandardScaler or RobustScaler."""
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()

    for col in cols:
        if col in df.columns:
            df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
    return df
