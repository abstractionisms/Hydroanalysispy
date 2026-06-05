"""Non-parametric changepoint tests for hydrologic time series."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def pettitt_test(series: pd.Series) -> Dict[str, Any]:
    """Detect one distributional changepoint using Pettitt's rank test."""
    clean = pd.Series(series).dropna().sort_index()
    n = len(clean)
    if n < 6:
        return {}

    values = clean.values.astype(float)
    ranks = pd.Series(values).rank().values
    u = np.array([2 * np.sum(ranks[: k + 1]) - (k + 1) * (n + 1) for k in range(n)])
    k = int(np.argmax(np.abs(u)))
    statistic = float(abs(u[k]))
    p_value = float(2 * np.exp((-6 * statistic**2) / (n**3 + n**2)))
    before = values[: k + 1]
    after = values[k + 1 :]
    change_label = clean.index[k].year if hasattr(clean.index[k], "year") else clean.index[k]

    return {
        "change_index": k,
        "change_point": change_label,
        "statistic": statistic,
        "p_value": min(max(p_value, 0.0), 1.0),
        "mean_before": float(np.mean(before)) if len(before) else float("nan"),
        "mean_after": float(np.mean(after)) if len(after) else float("nan"),
        "n_points": n,
    }
