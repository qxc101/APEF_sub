from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from .eval import spearman_corr
from .llm import OpenAIChatConfig, openai_chat
from .timeseries import normalize_for_prompt

import tqdm
import time

_FINAL_SCORE_RE = re.compile(r"FINAL_SCORE:\s*([+-]?\d+(?:\.\d+)?)")


def validate_policy_structure(policy: str) -> bool:
    required = ["METRICS:", "FORMULA:", "SCORING:", "DECISION:"]
    return all(r in policy for r in required)


def extract_policy_score(text: str) -> float:
    m = _FINAL_SCORE_RE.search(text)
    if not m:
        return -1.0
    try:
        return float(m.group(1))
    except ValueError:
        return -1.0


def apply_policy_to_single(
    *,
    policy: str,
    ground_truth: np.ndarray,
    candidate: np.ndarray,
    cfg: OpenAIChatConfig,
    ndigits: int = 4,
) -> tuple[float, str]:
    gt_avg, cand_avg = normalize_for_prompt(ground_truth, candidate, ndigits=ndigits)
    prompt = f"""
You have the following policy:

{policy}

Compute a single numeric "FinalScore" for the below two time series to check how well this series matches the ground truth. Follow the exact step listed below.

Ground Truth (10-day avg, normalized): {gt_avg}
Candidate Series (10-day avg, normalized): {cand_avg}

Steps:
1. Use the METRICS in the policy to compute intermediate values if needed.
2. Combine them using the FORMULA/SCORING into a single numeric "FinalScore".
3. No pairwise comparison—just produce the one score.

Output exactly:
FINAL_SCORE: <float>
""".strip()
    # time_start = time.time()
    raw = openai_chat(
        system="You are a domain expert. Always show numeric calculations explicitly.",
        user=prompt,
        cfg=cfg,
    )
    # time_end = time.time()
    # print(f"Policy application took {time_end - time_start:.1f} seconds.")
    return extract_policy_score(raw), raw


def build_policy_extraction_prompt(
    *,
    recent_history: list[dict[str, Any]],
    current_policy: str,
    current_train_corr: float,
) -> str:
    history_str = "Recent weight adjustment performance (most recent last):\n"
    for i, entry in enumerate(recent_history, 1):
        history_str += (
            f"\nCase {i}:\n"
            f"  - pair: {entry.get('pair')}\n"
            f"  - desired: {entry.get('desired')}\n"
            f"  - train_spearman: {entry.get('train_spearman')}\n"
            f"  - weights: {entry.get('weights')}\n"
        )

    prompt = f"""
We are extracting and refining a human-readable evaluation policy for comparing ecological time series predictions against ground truth.

We maintain a base metric with tunable weights. The weight-optimization history below shows which weight settings improved Spearman correlation with the target ranking.

Your job: produce an updated evaluation policy that a human can apply manually to compute a single numeric score for any candidate series vs the ground truth.

Constraints:
- Keep the policy concise but precise.
- You may add/remove at most ONE metric compared to the current policy.
- Use explicit mathematical formulas.
- Output must contain the required sections exactly.

{history_str}

Current policy:
{current_policy}

Current train correlation (Spearman): {current_train_corr}

Required sections and format EXACTLY:

METRICS:
[bulleted list of metric names + short definitions]

FORMULA:
[explicit formulas for each metric]

SCORING:
[how to combine into FinalScore; weights/points sum to 10]

DECISION:
[FinalScore definition; tie-break rule]
""".strip()
    return prompt


def extract_evaluation_policy(
    *,
    weight_history: list[dict[str, Any]],
    current_policy: str,
    current_train_corr: float,
    cfg: OpenAIChatConfig,
    n_recent: int = 5,
) -> tuple[str, str]:
    recent = weight_history[-n_recent:] if len(weight_history) >= n_recent else list(weight_history)
    prompt = build_policy_extraction_prompt(
        recent_history=recent,
        current_policy=current_policy,
        current_train_corr=current_train_corr,
    )
    raw = openai_chat(
        system="You are a domain expert in time series flux/hydrology data. Output only the updated policy in the required format.",
        user=prompt,
        cfg=cfg,
    )
    if not validate_policy_structure(raw):
        return current_policy, raw
    return raw, raw


@dataclass(frozen=True)
class PolicyEvalResult:
    spearman: float
    scores: list[float]
    raw_outputs: list[str]


def evaluate_policy_correlation(
    *,
    policy: str,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    target_scores: list[float],
    indices: list[int],
    cfg: OpenAIChatConfig,
    ndigits: int = 4,
) -> PolicyEvalResult:
    scores: list[float] = []
    raws: list[str] = []
    for idx in tqdm.tqdm(indices, desc="Evaluating policy correlation"):
        s, raw = apply_policy_to_single(
            policy=policy,
            ground_truth=ground_truth,
            candidate=predictions[idx],
            cfg=cfg,
            ndigits=ndigits,
        )
        scores.append(float(s))
        raws.append(raw)

    corr = spearman_corr([target_scores[i] for i in indices], scores) if len(indices) >= 2 else float("nan")
    return PolicyEvalResult(spearman=float(corr), scores=scores, raw_outputs=raws)

