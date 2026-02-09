from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:  # pragma: no cover
    spearmanr = None  # type: ignore


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if spearmanr is None:  # pragma: no cover
        # Very small fallback (no tie handling)
        ra = np.argsort(np.argsort(np.asarray(a)))
        rb = np.argsort(np.argsort(np.asarray(b)))
        return float(np.corrcoef(ra, rb)[0, 1])
    corr, _ = spearmanr(a, b)
    return float(corr)


@dataclass(frozen=True)
class PairwiseAccuracy:
    n_total: int
    n_correct: int
    accuracy: float


def pairwise_accuracy(
    comparisons: list[tuple[int, int, int]],
    scores: list[float],
) -> PairwiseAccuracy:
    correct = 0
    total = 0
    for a, b, winner in comparisons:
        if winner not in {1, 2}:
            continue
        if not (0 <= a < len(scores) and 0 <= b < len(scores)):
            continue
        sa = scores[a]
        sb = scores[b]
        pred = 1 if sa > sb else 2
        total += 1
        if pred == winner:
            correct += 1
    acc = (correct / total) if total else 0.0
    return PairwiseAccuracy(n_total=total, n_correct=correct, accuracy=float(acc))

