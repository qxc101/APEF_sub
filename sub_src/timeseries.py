from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .normalize import normalize_pair_zscore


def as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def ten_day_average(x: np.ndarray) -> list[float]:
    ts = as_1d(x)
    n = int(ts.shape[0])
    if n == 0:
        return []

    period = 10
    n_periods = n // period
    out: list[float] = []
    if n_periods > 0:
        reshaped = ts[: n_periods * period].reshape(n_periods, period)
        out.extend([float(v) for v in np.mean(reshaped, axis=1)])
    remainder = ts[n_periods * period :]
    if remainder.size:
        out.append(float(np.mean(remainder)))
    return out


def round_floats(values: Iterable[float], *, ndigits: int = 4) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(float("nan"))
        else:
            out.append(round(float(v), ndigits))
    return out


def normalize_for_prompt(
    ground_truth: np.ndarray, prediction: np.ndarray, *, ndigits: int = 4
) -> tuple[list[float], list[float]]:
    gt, pred, _ = normalize_pair_zscore(as_1d(ground_truth), as_1d(prediction))
    return round_floats(ten_day_average(gt), ndigits=ndigits), round_floats(ten_day_average(pred), ndigits=ndigits)

