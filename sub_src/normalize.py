from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationParams:
    mean: float
    scale: float


def zscore_params(reference: np.ndarray, *, eps: float = 1e-8) -> NormalizationParams:
    ref = np.asarray(reference, dtype=float)
    mean = float(np.mean(ref))
    scale = float(np.std(ref))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return NormalizationParams(mean=mean, scale=scale)


def apply_zscore(x: np.ndarray, params: NormalizationParams) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr - params.mean) / params.scale


def normalize_pair_zscore(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, NormalizationParams]:
    params = zscore_params(ground_truth, eps=eps)
    gt = apply_zscore(ground_truth, params)
    pred = apply_zscore(prediction, params)
    return gt, pred, params

