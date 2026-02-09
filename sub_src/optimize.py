from __future__ import annotations

import random
from typing import Any, Literal

import numpy as np

from dataclasses import dataclass

from .eval import mae, pairwise_accuracy, r2_score, rmse, spearman_corr
from .llm import OpenAIChatConfig, OpenAIParseError, OpenAIUnavailable, clamp_weights, suggest_weights_openai
from .metric import DomainMetricWeights, domain_metric


LLMMode = Literal["none", "openai"]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def random_weights(rng: random.Random) -> DomainMetricWeights:
    return DomainMetricWeights(
        peak_period_weight=rng.random(),
        derivative_weight=rng.random(),
        tolerance_days=rng.randint(1, 10),
        amplitude_weight=rng.random(),
    )


def propose_weights_heuristic(rng: random.Random, w: DomainMetricWeights) -> DomainMetricWeights:
    return clamp_weights(
        DomainMetricWeights(
            peak_period_weight=_clip01(w.peak_period_weight + rng.uniform(-0.15, 0.15)),
            derivative_weight=_clip01(w.derivative_weight + rng.uniform(-0.15, 0.15)),
            tolerance_days=int(np.clip(w.tolerance_days + rng.choice([-2, -1, 0, 1, 2]), 1, 30)),
            amplitude_weight=_clip01(w.amplitude_weight + rng.uniform(-0.15, 0.15)),
        )
    )


def compute_scores(
    gt: np.ndarray,
    preds: np.ndarray,
    weights: DomainMetricWeights,
    *,
    normalize: bool = True,
) -> list[float]:
    scores: list[float] = []
    for i in range(int(preds.shape[0])):
        s = domain_metric(gt, preds[i], weights, normalize=normalize)
        scores.append(float(s))
    return scores


def _pair_stats(
    gt: np.ndarray, a: np.ndarray, b: np.ndarray, weights: DomainMetricWeights, *, normalize: bool
) -> dict[str, Any]:
    def stats_one(x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=float).reshape(-1)
        d1 = np.diff(x)
        return {
            "r2": r2_score(gt, x),
            "rmse": rmse(gt, x),
            "mae": mae(gt, x),
            "peak_day": int(np.argmax(x)),
            "amplitude": float(np.max(x) - np.min(x)),
            "mean_abs_diff": float(np.mean(np.abs(d1))) if d1.size else 0.0,
            "score_current": float(domain_metric(gt, x, weights, normalize=normalize)),
            "score_peaks_only": float(
                domain_metric(
                    gt,
                    x,
                    DomainMetricWeights(
                        peak_period_weight=weights.peak_period_weight,
                        derivative_weight=0.0,
                        tolerance_days=weights.tolerance_days,
                        amplitude_weight=weights.amplitude_weight,
                    ),
                    normalize=normalize,
                )
            ),
            "score_deriv_only": float(
                domain_metric(
                    gt,
                    x,
                    DomainMetricWeights(
                        peak_period_weight=weights.peak_period_weight,
                        derivative_weight=1.0,
                        tolerance_days=weights.tolerance_days,
                        amplitude_weight=weights.amplitude_weight,
                    ),
                    normalize=normalize,
                )
            ),
        }

    return {"A": stats_one(a), "B": stats_one(b)}


@dataclass(frozen=True)
class OptimizeResult:
    best_weights: DomainMetricWeights
    best_val_objective: float
    history: list[dict[str, Any]]


def optimize_exp1_to_target_scores(
    gt: np.ndarray,
    preds: np.ndarray,
    target_scores: list[float],
    *,
    train_idx: list[int],
    val_idx: list[int],
    steps: int,
    seed: int,
    llm_mode: LLMMode,
    openai_cfg: OpenAIChatConfig | None,
    normalize: bool = True,
) -> OptimizeResult:
    rng = random.Random(seed)
    current = random_weights(rng)
    current_scores = compute_scores(gt, preds, current, normalize=normalize)

    def obj(indices: list[int], scores: list[float]) -> float:
        a = [target_scores[i] for i in indices]
        b = [scores[i] for i in indices]
        c = spearman_corr(a, b)
        return float(c) if np.isfinite(c) else float("-inf")

    current_obj = obj(train_idx, current_scores)
    current_val = obj(val_idx, current_scores) if val_idx else float("nan")

    best = current
    best_val = current_val if np.isfinite(current_val) else float("-inf")

    history: list[dict[str, Any]] = []
    history.append(
        {
            "step": 0,
            "accepted": True,
            "train_spearman": current_obj,
            "val_spearman": current_val,
            "weights": current.__dict__,
            "note": "init",
        }
    )

    for step in range(1, steps + 1):
        if len(train_idx) < 2:
            break
        i, j = rng.sample(train_idx, 2)
        desired = "A higher than B" if target_scores[i] > target_scores[j] else "A lower than B"

        score_a = float(domain_metric(gt, preds[i], current, normalize=normalize))
        score_b = float(domain_metric(gt, preds[j], current, normalize=normalize))
        already_ok = (score_a > score_b) if (desired == "A higher than B") else (score_a < score_b)

        candidate = current
        raw = ""
        note = "skip (already satisfies pair)" if already_ok else "proposed"
        if not already_ok:
            if llm_mode == "openai":
                if openai_cfg is None:
                    raise ValueError("openai_cfg is required when llm_mode='openai'")
                stats = _pair_stats(gt, preds[i], preds[j], current, normalize=normalize)
                try:
                    candidate, raw = suggest_weights_openai(
                        desired_relation=desired,
                        current_weights=current,
                        score_a=score_a,
                        score_b=score_b,
                        stats=stats,
                        cfg=openai_cfg,
                    )
                    note = "openai_suggest"
                except OpenAIUnavailable:
                    raise
                except OpenAIParseError as e:
                    candidate = propose_weights_heuristic(rng, current)
                    raw = e.raw_output
                    note = "openai_parse_error_fallback_heuristic"
                except Exception as e:
                    candidate = propose_weights_heuristic(rng, current)
                    raw = f"{type(e).__name__}: {e}"
                    note = "openai_error_fallback_heuristic"
            else:
                candidate = propose_weights_heuristic(rng, current)
                note = "heuristic_suggest"

        cand_scores = compute_scores(gt, preds, candidate, normalize=normalize)
        cand_obj = obj(train_idx, cand_scores)
        cand_val = obj(val_idx, cand_scores) if val_idx else float("nan")

        accepted = bool(cand_obj >= current_obj)
        if accepted:
            current = candidate
            current_scores = cand_scores
            current_obj = cand_obj
            current_val = cand_val
            if np.isfinite(cand_val) and cand_val > best_val:
                best = candidate
                best_val = cand_val

        history.append(
            {
                "step": step,
                "pair": [i, j],
                "desired": desired,
                "score_a": score_a,
                "score_b": score_b,
                "accepted": accepted,
                "train_spearman": cand_obj,
                "val_spearman": cand_val,
                "weights": candidate.__dict__,
                "note": note,
                "llm_raw": raw,
            }
        )

    return OptimizeResult(best_weights=best, best_val_objective=float(best_val), history=history)


def optimize_exp2_to_pairwise_preferences(
    gt: np.ndarray,
    preds: np.ndarray,
    *,
    train: list[tuple[int, int, int]],
    val: list[tuple[int, int, int]],
    steps: int,
    seed: int,
    llm_mode: LLMMode,
    openai_cfg: OpenAIChatConfig | None,
    normalize: bool = True,
) -> OptimizeResult:
    rng = random.Random(seed)
    current = random_weights(rng)

    current_scores = compute_scores(gt, preds, current, normalize=normalize)
    current_acc = pairwise_accuracy(train, current_scores).accuracy
    current_val = pairwise_accuracy(val, current_scores).accuracy if val else float("nan")

    best = current
    best_val = current_val if np.isfinite(current_val) else float("-inf")

    history: list[dict[str, Any]] = []
    history.append(
        {
            "step": 0,
            "accepted": True,
            "train_acc": current_acc,
            "val_acc": current_val,
            "weights": current.__dict__,
            "note": "init",
        }
    )

    for step in range(1, steps + 1):
        if not train:
            break
        a, b, winner = rng.choice(train)
        desired = "A higher than B" if winner == 1 else "A lower than B"

        score_a = float(domain_metric(gt, preds[a], current, normalize=normalize))
        score_b = float(domain_metric(gt, preds[b], current, normalize=normalize))
        already_ok = (score_a > score_b) if (winner == 1) else (score_a < score_b)

        candidate = current
        raw = ""
        note = "skip (already satisfies pair)" if already_ok else "proposed"
        if not already_ok:
            if llm_mode == "openai":
                if openai_cfg is None:
                    raise ValueError("openai_cfg is required when llm_mode='openai'")
                stats = _pair_stats(gt, preds[a], preds[b], current, normalize=normalize)
                try:
                    candidate, raw = suggest_weights_openai(
                        desired_relation=desired,
                        current_weights=current,
                        score_a=score_a,
                        score_b=score_b,
                        stats=stats,
                        cfg=openai_cfg,
                    )
                    note = "openai_suggest"
                except OpenAIUnavailable:
                    raise
                except OpenAIParseError as e:
                    candidate = propose_weights_heuristic(rng, current)
                    raw = e.raw_output
                    note = "openai_parse_error_fallback_heuristic"
                except Exception as e:
                    candidate = propose_weights_heuristic(rng, current)
                    raw = f"{type(e).__name__}: {e}"
                    note = "openai_error_fallback_heuristic"
            else:
                candidate = propose_weights_heuristic(rng, current)
                note = "heuristic_suggest"

        cand_scores = compute_scores(gt, preds, candidate, normalize=normalize)
        cand_acc = pairwise_accuracy(train, cand_scores).accuracy
        cand_val = pairwise_accuracy(val, cand_scores).accuracy if val else float("nan")

        accepted = bool(cand_acc >= current_acc)
        if accepted:
            current = candidate
            current_scores = cand_scores
            current_acc = cand_acc
            current_val = cand_val
            if np.isfinite(cand_val) and cand_val > best_val:
                best = candidate
                best_val = cand_val

        history.append(
            {
                "step": step,
                "pair": [a, b],
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
                "accepted": accepted,
                "train_acc": cand_acc,
                "val_acc": cand_val,
                "weights": candidate.__dict__,
                "note": note,
                "llm_raw": raw,
            }
        )

    return OptimizeResult(best_weights=best, best_val_objective=float(best_val), history=history)
