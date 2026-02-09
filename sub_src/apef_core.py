from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np

from .baselines import PRPConfig, baseline_correlations, baseline_rows
from .eval import spearman_corr
from .io_utils import write_csv, write_json, write_jsonl
from .llm import OpenAIChatConfig, OpenAIParseError, OpenAIUnavailable, suggest_weights_openai
from .logging_utils import setup_run_logger
from .metric import DomainMetricWeights, domain_metric
from .optimize import propose_weights_heuristic
from .policy import evaluate_policy_correlation, extract_evaluation_policy
from .timeseries import normalize_for_prompt

import time

LLMMode = Literal["none", "openai"]


@dataclass(frozen=True)
class APEFConfig:
    steps: int = 20
    warmup_steps: int = 10
    policy_extract_every: int = 1
    policy_eval_runs: int = 1
    policy_accept_frac: float = 0.6
    require_positive_first_policy: bool = True
    avoid_repeat_pairs: bool = True
    prompt_ndigits: int = 4


@dataclass(frozen=True)
class APEFRunResult:
    run_dir: Path
    best_weights: DomainMetricWeights
    best_weights_val_spearman: float
    best_policy: str | None
    best_policy_val_spearman: float


def _random_initial_weights(rng: random.Random) -> DomainMetricWeights:
    # Keep peak/deriv/amp weights summing to 1 (mirrors `src/policy_concrete.py` init).
    a = rng.random()
    b = rng.random()
    c = rng.random()
    s = a + b + c
    a /= s
    b /= s
    c /= s
    tol = rng.randint(1, 10)
    return DomainMetricWeights(
        peak_period_weight=float(a),
        derivative_weight=float(b),
        tolerance_days=int(tol),
        amplitude_weight=float(c),
    )


def _metric_scores(gt: np.ndarray, preds: np.ndarray, w: DomainMetricWeights, *, normalize: bool) -> list[float]:
    scores: list[float] = []
    for i in range(int(preds.shape[0])):
        scores.append(float(domain_metric(gt, preds[i], w, normalize=normalize)))
    return scores


def _spearman_on(indices: list[int], target: list[float], scores: list[float]) -> float:
    if len(indices) < 2:
        return float("nan")
    return spearman_corr([target[i] for i in indices], [scores[i] for i in indices])


def run_apef(
    *,
    exp_name: str,
    dataset_key: str,
    feature_name: str,
    scene: str | None,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    ts_labels: list[str],
    target_scores: list[float],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    llm_mode: LLMMode,
    openai_cfg: OpenAIChatConfig | None,
    openai_cfg_policy_extract: OpenAIChatConfig | None,
    openai_cfg_policy_score: OpenAIChatConfig | None,
    normalize: bool,
    prp_cfg: PRPConfig,
    out_base: Path,
    seed: int,
    cfg: APEFConfig,
) -> APEFRunResult:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scene_part = "" if scene is None else f"_{scene}"
    run_dir = out_base / f"{exp_name}_{dataset_key}{scene_part}_{timestamp}_seed{seed}"
    logger = setup_run_logger(run_dir, name=f"apef.{exp_name}.{dataset_key}.{scene or 'na'}.{seed}")

    logger.info(f"Starting {exp_name} | dataset={dataset_key} | feature={feature_name} | scene={scene}")
    logger.info(
        f"steps={cfg.steps} warmup={cfg.warmup_steps} policy_every={cfg.policy_extract_every} "
        f"policy_runs={cfg.policy_eval_runs} accept_frac={cfg.policy_accept_frac}"
    )
    logger.info(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    if llm_mode == "openai":
        logger.info(
            "openai models: "
            f"weights/prp={getattr(openai_cfg, 'model', None)} "
            f"policy_extract={getattr(openai_cfg_policy_extract, 'model', None)} "
            f"policy_score={getattr(openai_cfg_policy_score, 'model', None)}"
        )

    # -----------------
    # Baselines
    # -----------------
    prp_llm_cfg = openai_cfg if (prp_cfg.enabled and llm_mode == "openai") else None
    effective_prp_cfg = prp_cfg if prp_llm_cfg is not None else PRPConfig(enabled=False)
    if prp_cfg.enabled and prp_llm_cfg is None:
        logger.info("PRP baseline disabled (requires --llm openai).")

    baseline = baseline_rows(
        dataset_key=dataset_key,
        ground_truth=ground_truth,
        predictions=predictions,
        ts_labels=ts_labels,
        prp_cfg=effective_prp_cfg,
        prp_llm_cfg=prp_llm_cfg,
    )
    all_idx = list(range(int(predictions.shape[0])))
    base_corr = {
        "train": baseline_correlations(rows=baseline, target_scores=target_scores, indices=train_idx),
        "val": baseline_correlations(rows=baseline, target_scores=target_scores, indices=val_idx),
        "test": baseline_correlations(rows=baseline, target_scores=target_scores, indices=test_idx),
        "all": baseline_correlations(rows=baseline, target_scores=target_scores, indices=all_idx),
    }
    write_csv(
        run_dir / "baseline_metrics.csv",
        ["ts", "ts_id", "r2", "rmse", "mae", "nse", "tildeq", "prp_rank"],
        baseline,
    )
    write_json(run_dir / "baseline_correlations.json", {"spearman": base_corr})
    logger.info(f"Baseline spearman (val): {base_corr.get('val')}")

    # -----------------
    # Weight optimization + policy extraction
    # -----------------
    rng = random.Random(seed)
    current_w = _random_initial_weights(rng)
    current_scores = _metric_scores(ground_truth, predictions, current_w, normalize=normalize)
    current_train_corr = _spearman_on(train_idx, target_scores, current_scores)
    current_val_corr = _spearman_on(val_idx, target_scores, current_scores) if val_idx else float("nan")

    best_w = current_w
    best_w_val = current_val_corr if np.isfinite(current_val_corr) else float("-inf")

    weight_history: list[dict[str, Any]] = [
        {
            "step": 0,
            "accepted": True,
            "weights": current_w.__dict__,
            "train_spearman": current_train_corr,
            "val_spearman": current_val_corr,
            "note": "init",
        }
    ]

    best_policy: str | None = None
    best_policy_val = float("-inf")
    current_policy = "METRICS:\n\nFORMULA:\n\nSCORING:\n\nDECISION:\n"
    policy_history: list[dict[str, Any]] = []

    compared_pairs: set[tuple[int, int]] = set()
    step_done = 0
    while step_done < int(cfg.steps):
        if len(train_idx) < 2:
            break

        i, j = rng.sample(train_idx, 2)
        if cfg.avoid_repeat_pairs and ((i, j) in compared_pairs or (j, i) in compared_pairs):
            continue
        compared_pairs.add((i, j))
        step_done += 1

        desired = "A higher than B" if target_scores[i] > target_scores[j] else "A lower than B"
        score_a = float(domain_metric(ground_truth, predictions[i], current_w, normalize=normalize))
        score_b = float(domain_metric(ground_truth, predictions[j], current_w, normalize=normalize))

        cand_w = current_w
        llm_raw = ""
        note = ""

        if llm_mode == "openai":
            if openai_cfg is None:
                raise ValueError("openai_cfg is required when llm_mode='openai'")
            try:
                gt_avg, a_avg = normalize_for_prompt(ground_truth, predictions[i], ndigits=cfg.prompt_ndigits)
                _gt_avg2, b_avg = normalize_for_prompt(ground_truth, predictions[j], ndigits=cfg.prompt_ndigits)
                cand_w, llm_raw = suggest_weights_openai(
                    desired_relation=desired,
                    current_weights=current_w,
                    score_a=score_a,
                    score_b=score_b,
                    stats={
                        "pair": [i, j],
                        "target_scores": [target_scores[i], target_scores[j]],
                        "ground_truth_avg": gt_avg,
                        "pred_a_avg": a_avg,
                        "pred_b_avg": b_avg,
                    },
                    cfg=openai_cfg,
                )
                note = "openai_suggest"
            except OpenAIUnavailable:
                raise
            except OpenAIParseError as e:
                cand_w = propose_weights_heuristic(rng, current_w)
                llm_raw = e.raw_output
                note = "openai_parse_error_fallback_heuristic"
            except Exception as e:
                cand_w = propose_weights_heuristic(rng, current_w)
                llm_raw = f"{type(e).__name__}: {e}"
                note = "openai_error_fallback_heuristic"
        else:
            cand_w = propose_weights_heuristic(rng, current_w)
            note = "heuristic_suggest"

        cand_scores = _metric_scores(ground_truth, predictions, cand_w, normalize=normalize)
        cand_train_corr = _spearman_on(train_idx, target_scores, cand_scores)
        cand_val_corr = _spearman_on(val_idx, target_scores, cand_scores) if val_idx else float("nan")

        accepted = bool(np.isfinite(cand_train_corr) and (not np.isfinite(current_train_corr) or cand_train_corr >= current_train_corr))
        if accepted:
            current_w = cand_w
            current_scores = cand_scores
            current_train_corr = cand_train_corr
            current_val_corr = cand_val_corr
            if np.isfinite(cand_val_corr) and cand_val_corr > best_w_val:
                best_w = cand_w
                best_w_val = cand_val_corr

        weight_history.append(
            {
                "step": step_done,
                "pair": [i, j],
                "desired": desired,
                "score_a": score_a,
                "score_b": score_b,
                "accepted": accepted,
                "weights": cand_w.__dict__,
                "train_spearman": cand_train_corr,
                "val_spearman": cand_val_corr,
                "note": note,
                "llm_raw": llm_raw,
            }
        )

        logger.info(
            f"[weights] step={step_done}/{cfg.steps} train={cand_train_corr:.3f} "
            f"val={(cand_val_corr if np.isfinite(cand_val_corr) else float('nan')):.3f} "
            f"accepted={accepted} best_policy_val={(best_policy_val if np.isfinite(best_policy_val) else float('nan')):.3f}"
        )

        # Policy extraction + validation
        if step_done < int(cfg.warmup_steps):
            continue
        if cfg.policy_extract_every <= 0:
            continue
        if ((step_done - int(cfg.warmup_steps)) % int(cfg.policy_extract_every)) != 0:
            continue
        if llm_mode != "openai" or openai_cfg is None:
            continue
        if not val_idx:
            continue
        policy_extract_cfg = openai_cfg_policy_extract or openai_cfg
        policy_score_cfg = openai_cfg_policy_score or openai_cfg
        time_start = time.time()
        new_policy, raw_policy = extract_evaluation_policy(
            weight_history=weight_history,
            current_policy=current_policy,
            current_train_corr=current_train_corr,
            cfg=policy_extract_cfg,
            n_recent=5,
        )
        time_end = time.time()
        print(f"Policy extraction took {time_end - time_start:.1f} seconds.")
        if new_policy == current_policy:
            policy_history.append(
                {
                    "step": step_done,
                    "accepted": False,
                    "note": "policy_invalid_or_unchanged",
                    "raw": raw_policy,
                }
            )
            continue

        best_before = best_policy_val
        eval_spearmans: list[float] = []
        for run_id in range(int(cfg.policy_eval_runs)):
            
            ev = evaluate_policy_correlation(
                policy=new_policy,
                ground_truth=ground_truth,
                predictions=predictions,
                target_scores=target_scores,
                indices=val_idx,
                cfg=policy_score_cfg,
                ndigits=cfg.prompt_ndigits,
            )
            eval_spearmans.append(float(ev.spearman))

            score_rows = []
            for local_i, ts_i in enumerate(val_idx):
                score_rows.append(
                    {
                        "ts": ts_labels[ts_i] if ts_i < len(ts_labels) else f"TS {ts_i}",
                        "ts_id": ts_i,
                        "target_score": float(target_scores[ts_i]),
                        "policy_score": float(ev.scores[local_i]),
                        "raw": ev.raw_outputs[local_i],
                    }
                )
            write_jsonl(run_dir / "policy_score_logs" / f"step{step_done:03d}_run{run_id}.jsonl", score_rows)

        improved = [v for v in eval_spearmans if np.isfinite(v) and v > best_before]
        frac = (len(improved) / len(eval_spearmans)) if eval_spearmans else 0.0
        avg_val = float(np.nanmean(eval_spearmans)) if eval_spearmans else float("nan")

        accept = bool(frac >= float(cfg.policy_accept_frac))
        if cfg.require_positive_first_policy and best_policy is None:
            accept = accept and (avg_val > 0)

        if accept:
            best_policy = new_policy
            best_policy_val = avg_val
            current_policy = new_policy
            (run_dir / "best_policy.txt").write_text(best_policy + "\n", encoding="utf-8")
            logger.info(f"[policy] ACCEPT step={step_done} val_avg={avg_val:.3f} evals={eval_spearmans}")
        else:
            logger.info(f"[policy] reject step={step_done} val_avg={avg_val:.3f} evals={eval_spearmans}")

        policy_history.append(
            {
                "step": step_done,
                "accepted": accept,
                "val_spearman_runs": eval_spearmans,
                "val_spearman_avg": avg_val,
                "frac_improved": frac,
                "best_policy_val_before": best_before,
                "best_policy_val_after": best_policy_val,
                "policy": new_policy,
                "raw": raw_policy,
            }
        )

    write_jsonl(run_dir / "weight_history.jsonl", weight_history)
    write_jsonl(run_dir / "policy_history.jsonl", policy_history)

    final_weight_scores = _metric_scores(ground_truth, predictions, best_w, normalize=normalize)
    weight_corrs = {
        "train": _spearman_on(train_idx, target_scores, final_weight_scores),
        "val": _spearman_on(val_idx, target_scores, final_weight_scores),
        "test": _spearman_on(test_idx, target_scores, final_weight_scores),
        "all": _spearman_on(list(range(int(predictions.shape[0]))), target_scores, final_weight_scores),
    }

    # Optional final policy evaluation
    policy_corrs: dict[str, float] = {}
    if best_policy is not None and llm_mode == "openai" and openai_cfg is not None:
        policy_score_cfg = openai_cfg_policy_score or openai_cfg

        def eval_split(name: str, idxs: list[int]) -> None:
            if not idxs:
                policy_corrs[name] = float("nan")
                return
            ev = evaluate_policy_correlation(
                policy=best_policy,
                ground_truth=ground_truth,
                predictions=predictions,
                target_scores=target_scores,
                indices=idxs,
                cfg=policy_score_cfg,
                ndigits=cfg.prompt_ndigits,
            )
            policy_corrs[name] = float(ev.spearman)
            rows = []
            for local_i, ts_i in enumerate(idxs):
                rows.append(
                    {
                        "ts": ts_labels[ts_i] if ts_i < len(ts_labels) else f"TS {ts_i}",
                        "ts_id": ts_i,
                        "target_score": float(target_scores[ts_i]),
                        "policy_score": float(ev.scores[local_i]),
                        "raw": ev.raw_outputs[local_i],
                    }
                )
            write_jsonl(run_dir / "final_policy_scores" / f"{name}.jsonl", rows)

        eval_split("val", val_idx)
        eval_split("test", test_idx)

    summary = {
        "exp": exp_name,
        "dataset": dataset_key,
        "feature": feature_name,
        "scene": scene,
        "seed": seed,
        "openai_models": (
            None
            if llm_mode != "openai"
            else {
                "weights_prp": getattr(openai_cfg, "model", None),
                "policy_extract": getattr(openai_cfg_policy_extract, "model", None),
                "policy_score": getattr(openai_cfg_policy_score, "model", None),
            }
        ),
        "config": cfg.__dict__,
        "normalize": normalize,
        "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
        "baselines_spearman": base_corr,
        "best_weights": best_w.__dict__,
        "best_weights_val_spearman": float(best_w_val),
        "weights_spearman": weight_corrs,
        "has_best_policy": best_policy is not None,
        "best_policy_val_spearman": float(best_policy_val),
        "policy_spearman": policy_corrs,
    }
    write_json(run_dir / "summary.json", summary)

    score_rows = []
    for i, label in enumerate(ts_labels):
        score_rows.append(
            {
                "ts": label,
                "ts_id": i,
                "target_score": float(target_scores[i]),
                "metric_score_best_weights": float(final_weight_scores[i]),
            }
        )
    write_csv(run_dir / "scores.csv", ["ts", "ts_id", "target_score", "metric_score_best_weights"], score_rows)

    logger.info(f"Done. run_dir={run_dir}")
    return APEFRunResult(
        run_dir=run_dir,
        best_weights=best_w,
        best_weights_val_spearman=float(best_w_val),
        best_policy=best_policy,
        best_policy_val_spearman=float(best_policy_val),
    )
