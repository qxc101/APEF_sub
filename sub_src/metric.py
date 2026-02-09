from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normalize import normalize_pair_zscore

try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:  # pragma: no cover
    find_peaks = None  # type: ignore


@dataclass(frozen=True)
class DomainMetricWeights:
    peak_period_weight: float
    derivative_weight: float
    tolerance_days: int
    amplitude_weight: float


@dataclass(frozen=True)
class DomainMetricDetails:
    score: float
    start_day: int
    end_day: int
    within_score: float
    out1_score: float
    out2_score: float


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def _peak_window_from_threshold(
    ts: np.ndarray,
    *,
    frac: float = 0.2,
    smooth_window: int = 7,
    min_width: int = 30,
) -> tuple[int, int]:
    """Return (start, end) indices for the main-peak season/window.

    This avoids assuming a sustained monotonic rise/fall (which fails for some datasets).
    """
    x = np.asarray(ts, dtype=float)
    n = int(x.shape[0])
    if n < 2:
        return 0, max(0, n - 1)

    xs = _moving_average(x, smooth_window)
    peak_idx = int(np.argmax(xs))
    peak_val = float(xs[peak_idx])

    baseline = float(np.percentile(xs, 5))
    if not np.isfinite(baseline):
        baseline = float(np.min(xs))
    if not np.isfinite(peak_val):
        peak_val = float(np.max(xs))

    thresh = baseline + frac * (peak_val - baseline)

    left = 0
    for i in range(peak_idx, -1, -1):
        if xs[i] <= thresh:
            left = i
            break

    right = n - 1
    for i in range(peak_idx, n):
        if xs[i] <= thresh:
            right = i
            break

    if right - left < min_width:
        half = max(1, min_width // 2)
        left = max(0, peak_idx - half)
        right = min(n - 1, peak_idx + half)

    if left >= right:
        left = max(0, peak_idx - 1)
        right = min(n - 1, peak_idx + 1)

    return int(left), int(right)


def _top_peaks(ts: np.ndarray, *, max_peaks: int = 10) -> np.ndarray:
    x = np.asarray(ts, dtype=float)
    if find_peaks is None:
        # Simple local maxima fallback
        peaks = [i for i in range(1, len(x) - 1) if x[i - 1] < x[i] >= x[i + 1]]
        peaks = np.array(peaks, dtype=int)
    else:
        peaks, _ = find_peaks(x)
        peaks = np.asarray(peaks, dtype=int)

    if peaks.size == 0:
        return peaks
    if peaks.size <= max_peaks:
        return np.sort(peaks)

    top = np.argsort(x[peaks])[-max_peaks:]
    return np.sort(peaks[top])


def _peak_similarity(
    true_ts: np.ndarray,
    pred_ts: np.ndarray,
    *,
    tolerance_days: int,
    amplitude_weight: float,
    max_peaks: int = 10,
) -> float:
    true_ts = np.asarray(true_ts, dtype=float)
    pred_ts = np.asarray(pred_ts, dtype=float)

    tolerance = max(1, int(tolerance_days))
    amp_w = _clip01(float(amplitude_weight))

    true_peaks = _top_peaks(true_ts, max_peaks=max_peaks)
    pred_peaks = _top_peaks(pred_ts, max_peaks=max_peaks)

    if true_peaks.size == 0 and pred_peaks.size == 0:
        return 1.0
    if true_peaks.size == 0 or pred_peaks.size == 0:
        return 0.0

    used_pred: set[int] = set()
    matched = 0
    total = 0.0

    for tp in true_peaks:
        best_dp = None
        best_dt = None
        for dp in pred_peaks:
            if int(dp) in used_pred:
                continue
            dt = abs(int(tp) - int(dp))
            if dt <= tolerance and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_dp = int(dp)
        if best_dp is None or best_dt is None:
            continue

        time_score = 1.0 - (best_dt / tolerance)
        denom = max(1e-8, max(abs(float(true_ts[tp])), abs(float(pred_ts[best_dp]))))
        amp_diff = abs(float(true_ts[tp]) - float(pred_ts[best_dp]))
        amp_score = 1.0 - (amp_diff / denom)

        combined = (1.0 - amp_w) * time_score + amp_w * amp_score
        total += _clip01(float(combined))
        matched += 1
        used_pred.add(best_dp)

    precision = matched / float(pred_peaks.size) if pred_peaks.size else 0.0
    recall = matched / float(true_peaks.size) if true_peaks.size else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    normalized_total = total / float(true_peaks.size) if true_peaks.size else 0.0

    return _clip01((f1 + normalized_total) / 2.0)


def _derivative_similarity(ts1: np.ndarray, ts2: np.ndarray, *, order: int) -> float:
    a = np.asarray(ts1, dtype=float)
    b = np.asarray(ts2, dtype=float)
    if a.size < (order + 2) or b.size < (order + 2):
        return 1.0

    da = np.diff(a, n=order)
    db = np.diff(b, n=order)
    if da.size == 0 or db.size == 0:
        return 1.0

    mse = float(np.mean((da - db) ** 2))
    energy = float(0.5 * (np.mean(da**2) + np.mean(db**2)))
    if not np.isfinite(energy) or energy <= 1e-12:
        return 1.0 if mse <= 1e-12 else 0.0

    rel = mse / energy
    return float(1.0 / (1.0 + rel))


def _segment_score(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    derivative_weight: float,
    tolerance_days: int,
    amplitude_weight: float,
) -> float:
    d_w = _clip01(float(derivative_weight))

    peak_sim = _peak_similarity(
        gt,
        pred,
        tolerance_days=tolerance_days,
        amplitude_weight=amplitude_weight,
    )
    d1 = _derivative_similarity(gt, pred, order=1)
    d2 = _derivative_similarity(gt, pred, order=2)
    deriv_sim = 0.5 * (d1 + d2)

    return (1.0 - d_w) * peak_sim + d_w * deriv_sim


def domain_metric(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    weights: DomainMetricWeights,
    *,
    normalize: bool = True,
    peak_window_frac: float = 0.2,
    peak_window_smooth: int = 7,
    peak_window_min_width: int = 30,
    return_details: bool = False,
) -> float | DomainMetricDetails:
    gt = np.asarray(ground_truth, dtype=float).reshape(-1)
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    if normalize:
        gt, pred, _ = normalize_pair_zscore(gt, pred)

    start, end = _peak_window_from_threshold(
        gt,
        frac=float(peak_window_frac),
        smooth_window=int(peak_window_smooth),
        min_width=int(peak_window_min_width),
    )

    within_gt = gt[start : end + 1]
    within_pred = pred[start : end + 1]
    out1_gt = gt[:start]
    out1_pred = pred[:start]
    out2_gt = gt[end + 1 :]
    out2_pred = pred[end + 1 :]

    within = _segment_score(
        within_gt,
        within_pred,
        derivative_weight=weights.derivative_weight,
        tolerance_days=weights.tolerance_days,
        amplitude_weight=weights.amplitude_weight,
    )
    out1 = _segment_score(
        out1_gt,
        out1_pred,
        derivative_weight=weights.derivative_weight,
        tolerance_days=weights.tolerance_days,
        amplitude_weight=weights.amplitude_weight,
    )
    out2 = _segment_score(
        out2_gt,
        out2_pred,
        derivative_weight=weights.derivative_weight,
        tolerance_days=weights.tolerance_days,
        amplitude_weight=weights.amplitude_weight,
    )

    ppw = _clip01(float(weights.peak_period_weight))
    score = ppw * within + (1.0 - ppw) * 0.5 * (out1 + out2)
    score = _clip01(score)

    if return_details:
        return DomainMetricDetails(
            score=score,
            start_day=int(start),
            end_day=int(end),
            within_score=float(within),
            out1_score=float(out1),
            out2_score=float(out2),
        )
    return score

