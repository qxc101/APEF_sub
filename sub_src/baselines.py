from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import numpy as np

from .eval import mae, r2_score, rmse, spearman_corr
from .llm import OpenAIChatConfig, compare_pair_openai
from .timeseries import normalize_for_prompt


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom <= 1e-12:
        return 0.0
    return 1.0 - float(np.sum((y_pred - y_true) ** 2)) / denom


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


@dataclass(frozen=True)
class PRPConfig:
    enabled: bool = True
    k: int = 2
    ndigits: int = 4
    cache_dir: Path | None = None


def _prp_cache_path(dataset_key: str, *, cfg: PRPConfig, llm_cfg: OpenAIChatConfig) -> Path | None:
    if cfg.cache_dir is None:
        return None
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"prp_{dataset_key}_k{cfg.k}_nd{cfg.ndigits}_{_sanitize(llm_cfg.model)}.json"
    return cfg.cache_dir / name


def prp_sliding_k(
    *,
    dataset_key: str,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    llm_cfg: OpenAIChatConfig,
    cfg: PRPConfig,
    force: bool = False,
) -> list[int] | None:
    if not cfg.enabled:
        return None
    n = int(predictions.shape[0])
    if n < 2:
        return list(range(n))

    cache_path = _prp_cache_path(dataset_key, cfg=cfg, llm_cfg=llm_cfg)
    if cache_path is not None and cache_path.exists() and not force:
        try:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            ranks = obj.get("ranking_positions")
            if isinstance(ranks, list) and len(ranks) == n:
                return [int(x) for x in ranks]
        except Exception:
            pass

    gt_avg, _ = normalize_for_prompt(ground_truth, ground_truth, ndigits=cfg.ndigits)
    pred_avgs = [normalize_for_prompt(ground_truth, predictions[i], ndigits=cfg.ndigits)[1] for i in range(n)]

    ranked_indices = list(range(n))

    def compare_and_swap(i: int, j: int) -> None:
        choice, _raw = compare_pair_openai(
            ground_truth_avg=gt_avg,
            pred_a_avg=pred_avgs[ranked_indices[i]],
            pred_b_avg=pred_avgs[ranked_indices[j]],
            cfg=llm_cfg,
        )
        if choice == "B":
            ranked_indices[i], ranked_indices[j] = ranked_indices[j], ranked_indices[i]

    for _pass in range(int(cfg.k)):
        for i in range(n - 1):
            compare_and_swap(i, i + 1)

    ranking_positions = [0] * n
    for rank, idx in enumerate(ranked_indices):
        ranking_positions[idx] = n - rank - 1

    if cache_path is not None:
        payload = {
            "dataset": dataset_key,
            "k": cfg.k,
            "ndigits": cfg.ndigits,
            "model": llm_cfg.model,
            "ranking_positions": ranking_positions,
        }
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ranking_positions


def compute_tildeq(gt: np.ndarray, pred: np.ndarray) -> float | None:
    try:
        import torch

        from .tildeq import tildeq_loss
    except Exception:
        return None

    pred_t = torch.tensor(np.asarray(pred, dtype=float), dtype=torch.float32).view(1, 1, -1)
    gt_t = torch.tensor(np.asarray(gt, dtype=float), dtype=torch.float32).view(1, 1, -1)
    return float(tildeq_loss(pred_t, gt_t).item())


def baseline_rows(
    *,
    dataset_key: str,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    ts_labels: list[str],
    prp_cfg: PRPConfig,
    prp_llm_cfg: OpenAIChatConfig | None,
) -> list[dict[str, Any]]:
    n = int(predictions.shape[0])
    prp_ranks: list[int] | None = None
    if prp_cfg.enabled:
        if prp_llm_cfg is None:
            raise ValueError("prp_llm_cfg is required when PRP is enabled")
        prp_ranks = prp_sliding_k(
            dataset_key=dataset_key,
            ground_truth=ground_truth,
            predictions=predictions,
            llm_cfg=prp_llm_cfg,
            cfg=prp_cfg,
        )

    rows: list[dict[str, Any]] = []
    for i in range(n):
        pred = predictions[i]
        row: dict[str, Any] = {
            "ts": ts_labels[i] if i < len(ts_labels) else f"TS {i}",
            "ts_id": i,
            "r2": r2_score(ground_truth, pred),
            "rmse": rmse(ground_truth, pred),
            "mae": mae(ground_truth, pred),
            "nse": nse(ground_truth, pred),
            "tildeq": compute_tildeq(ground_truth, pred),
        }
        if prp_ranks is not None:
            row["prp_rank"] = int(prp_ranks[i])
        rows.append(row)
    return rows


def baseline_correlations(
    *,
    rows: list[dict[str, Any]],
    target_scores: list[float],
    indices: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = ["r2", "rmse", "mae", "nse", "tildeq", "prp_rank"]
    for k in keys:
        xs: list[float] = []
        ys: list[float] = []
        for i in indices:
            if i < 0 or i >= len(rows):
                continue
            v = rows[i].get(k)
            if v is None:
                continue
            try:
                xs.append(float(v))
            except Exception:
                continue
            ys.append(float(target_scores[i]))
        out[k] = spearman_corr(xs, ys) if len(xs) >= 2 else float("nan")
    return out

