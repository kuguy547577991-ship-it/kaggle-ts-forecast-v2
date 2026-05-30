"""Integration test fixtures for the TS forecast project."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Generate synthetic store-item sales data."""
    dates = pd.date_range("2013-01-01", periods=100, freq="D")
    records = []
    for store in ["A", "B"]:
        for item in ["X", "Y"]:
            base = np.random.randint(10, 50)
            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "store": store,
                    "item": item,
                    "sales": base + i * 0.1 + np.random.normal(0, 2),
                })
    return pd.DataFrame(records)
