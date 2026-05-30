"""FastAPI inference server for Kaggle TS forecast models.

Usage:
    uvicorn serving.api:app --reload --port 8000
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Kaggle TS Forecast API", version="0.1.0")

MODEL_DIR = Path("data/06_models")
_models_cache: dict[str, Any] = {}


class ForecastInput(BaseModel):
    store: str
    item: str
    sales_lag_1: float
    sales_lag_2: float
    sales_lag_3: float
    sales_lag_7: float
    sales_rolling_mean_7: float
    sales_rolling_mean_14: float
    year: int
    month: int
    dayofweek: int
    is_weekend: int


class ForecastOutput(BaseModel):
    predicted_sales: float
    model_used: str


def _load_model(name: str = "lightgbm") -> Any:
    if name in _models_cache:
        return _models_cache[name]
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model {name} not found")
    with open(path, "rb") as f:
        _models_cache[name] = pickle.load(f)
    return _models_cache[name]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ForecastOutput)
def predict(data: ForecastInput) -> ForecastOutput:
    model = _load_model("lightgbm")
    input_df = pd.DataFrame([data.model_dump()])
    prediction = float(model.predict(input_df.values)[0])
    return ForecastOutput(predicted_sales=prediction, model_used="lightgbm")
