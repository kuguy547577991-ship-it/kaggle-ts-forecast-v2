"""Time series visualization utilities — adapted from kaggle-ts-forecast utils/plots.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str = "Time Series",
    group_column: str | None = None,
) -> plt.Figure:
    """Plot a time series with optional grouping."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if group_column and group_column in df.columns:
        for name, group in df.groupby(group_column):
            ax.plot(group[date_column], group[value_column], label=str(name), alpha=0.7)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.plot(df[date_column], df[value_column])

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_column)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig


def plot_seasonal_decomposition(
    result, title: str = "Seasonal Decomposition",
) -> plt.Figure:
    """Plot statsmodels seasonal decomposition result."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot feature importance as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df = importance_df.sort_values("importance", ascending=True)
    bars = ax.barh(importance_df["feature"], importance_df["importance"])
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_acf_pacf(
    series: np.ndarray, lags: int = 50, title: str = "ACF/PACF",
) -> plt.Figure:
    """Plot ACF and PACF for a time series."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_cv_scores(
    scores: dict[str, list[float]], title: str = "Cross-Validation Scores",
) -> plt.Figure:
    """Plot cross-validation scores across folds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, fold_scores in scores.items():
        ax.plot(range(1, len(fold_scores) + 1), fold_scores, marker="o", label=model_name)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model",
) -> plt.Figure:
    """Plot residuals analysis: scatter, histogram, time series, Q-Q plot."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title(f"{model_name}: Residuals vs Predicted")

    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(x=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_title("Residual Distribution")

    axes[1, 0].scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals Over Time")

    import scipy.stats as stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")

    plt.tight_layout()
    return fig
