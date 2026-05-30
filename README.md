# Kaggle Time Series Forecast v2

Store-item sales forecasting pipeline — a hand-written refactor of a Kedro project, kept lightweight and config-driven.

## Quick Start

```bash
# Install
make install

# Run pipeline
make train

# Run tests
make test

# Lint
make lint
```

## Project Structure

```
pipelines/ts_forecast/      # Core pipeline
├── config/
│   └── base.yaml           # All hyperparameters (models, features, split, eval)
├── pipeline.py             # Entry point — orchestrates the 4 steps
└── nodes/                  # Pipeline step implementations
    ├── preprocessing.py    #   Load → fill missing → label encode
    ├── feature_engineering.py  #   Datetime + lag + rolling + diff + EWM
    ├── training.py         #   LightGBM / XGBoost / RF / Ridge (CV)
    └── evaluation.py       #   Metrics + feature importance + plots

lib/                        # Reusable utilities
├── metrics.py              #   MASE, SMAPE, and other time-series metrics
└── viz.py                  #   ACF/PACF, seasonal decomposition, residuals

serving/                    # FastAPI inference server
orchestration/              # Airflow & Prefect scheduled training
monitoring/                 # Data quality (Great Expectations) + drift (Evidently)
tests/                      # Shared fixtures & integration tests
```

## Pipeline Flow

1. **Preprocessing** — Load CSV, parse dates, forward-fill missing values, label-encode categorical columns
2. **Feature Engineering** — Datetime features (year/month/dayofweek/...), lag features (1/2/3/7/14/28), rolling statistics (mean/std/min/max over 7/14/30), differencing, and EWM
3. **Training** — LightGBM & XGBoost with TimeSeriesSplit CV, Random Forest & Ridge as baselines. MLflow tracks everything.
4. **Evaluation** — MAE / RMSE / MAPE / R² across all 4 models. Outputs interactive Plotly prediction charts and feature importance.

## Configuration

All tuning lives in [base.yaml](pipelines/ts_forecast/config/base.yaml). Set personal overrides in `config/local.yaml` (gitignored).

```bash
py pipelines/ts_forecast/pipeline.py           # default config
py pipelines/ts_forecast/pipeline.py -c local  # with local overrides
```

## Tech Stack

| Category | Tools |
|---|---|
| Models | LightGBM, XGBoost, Random Forest, Ridge |
| Features | Lag, rolling, differencing, EWM, datetime |
| Experiment Tracking | MLflow |
| Tuning | Optuna |
| Serving | FastAPI, Pydantic |
| Orchestration | Airflow, Prefect |
| Monitoring | Great Expectations, Evidently |
| Viz | Plotly, Matplotlib, Seaborn |

## Requirements

Python >= 3.10. See [pyproject.toml](pyproject.toml) for full dependency list.
