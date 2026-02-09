from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import os
import re
from typing import Any

from .metric import DomainMetricWeights


class OpenAIUnavailable(RuntimeError):
    pass


class OpenAIParseError(RuntimeError):
    def __init__(self, message: str, *, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _iter_brace_objects(text: str) -> list[str]:
    s = _strip_code_fences(text)
    objs: list[str] = []
    n = len(s)
    for start in range(n):
        if s[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        for i in range(start, n):
            ch = s[i]

            if escape:
                escape = False
                continue

            if ch == "\\" and in_string:
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(s[start : i + 1])
                    break

    return objs


def _repair_jsonish(s: str) -> str:
    t = _strip_code_fences(s).strip().rstrip(";")
    # Common model slipups
    t = t.replace(";", ",")
    t = re.sub(r",(\s*[}\]])", r"\1", t)  # trailing commas
    t = t.replace("None", "null").replace("True", "true").replace("False", "false")
    # Quote unquoted keys: {a: 1} / , a: 1
    t = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', t)
    # Use double quotes (we only expect numeric values here)
    t = t.replace("'", '"')
    # Insert missing commas between values and next key: 0.1 "k":
    t = re.sub(r'([0-9eE+\-\.])\s*(")', r"\1, \2", t)
    return t


def _extract_weights_object(text: str) -> dict[str, Any]:
    s = _strip_code_fences(text)
    candidates = _iter_brace_objects(s)
    if not candidates and s.startswith("{") and s.endswith("}"):
        candidates = [s]

    required = {"peak_period_weight", "derivative_weight", "tolerance_days", "amplitude_weight"}
    last_err: Exception | None = None
    last_obj: dict[str, Any] | None = None
    for cand in candidates:
        for attempt in (cand, _repair_jsonish(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    last_obj = obj
                    if required.intersection(obj.keys()):
                        return obj
            except Exception as e:
                last_err = e
            try:
                obj2 = ast.literal_eval(attempt)
                if isinstance(obj2, dict):
                    last_obj = obj2
                    if required.intersection(obj2.keys()):
                        return obj2
            except Exception as e:
                last_err = e

    if last_obj is not None:
        raise ValueError(f"Parsed a JSON object but it did not contain weight keys: {sorted(last_obj.keys())}")
    if last_err is not None:
        raise ValueError(f"Could not parse weights JSON: {type(last_err).__name__}: {last_err}") from last_err
    raise ValueError("No JSON object found in LLM output")


def _as_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).strip())


def _as_int(v: Any) -> int:
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(round(v))
    return int(round(float(str(v).strip())))


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def clamp_weights(w: DomainMetricWeights) -> DomainMetricWeights:
    peak = _clip(float(w.peak_period_weight), 0.0, 1.0)
    deriv = _clip(float(w.derivative_weight), 0.0, 1.0)
    amp = _clip(float(w.amplitude_weight), 0.0, 1.0)
    s = peak + deriv + amp
    if s <= 1e-8:
        peak, deriv, amp = 1 / 3, 1 / 3, 1 / 3
    else:
        peak /= s
        deriv /= s
        amp /= s
    return DomainMetricWeights(
        peak_period_weight=peak,
        derivative_weight=deriv,
        tolerance_days=int(max(1, min(30, int(w.tolerance_days)))),
        amplitude_weight=amp,
    )


@dataclass(frozen=True)
class OpenAIChatConfig:
    model: str = "gpt-5-nano"
    temperature: float | None = None


def _openai_chat_complete(
    messages: list[dict[str, str]],
    *,
    cfg: OpenAIChatConfig,
    json_mode: bool = False,
) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OpenAIUnavailable(
            "OpenAI python package not available. Install it in your environment (e.g. `pip install openai`)."
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    # Prefer structured JSON output when supported by the installed OpenAI SDK.
    kwargs: dict[str, Any] = {"model": cfg.model, "messages": messages}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except TypeError as e:
            last_err = e
            # Older SDKs may not support `response_format`.
            if "response_format" in kwargs:
                kwargs = dict(kwargs)
                kwargs.pop("response_format", None)
                continue
            raise
        except Exception as e:
            last_err = e
            msg = str(e)
            # Some models reject `response_format` (API-level).
            if "response_format" in kwargs and "response_format" in msg:
                kwargs = dict(kwargs)
                kwargs.pop("response_format", None)
                continue
            # Some models reject setting temperature (only default allowed).
            if "temperature" in kwargs and ("Only the default (1) value is supported" in msg or "does not support" in msg):
                kwargs = dict(kwargs)
                kwargs.pop("temperature", None)
                continue
            raise
    else:  # pragma: no cover
        raise last_err or RuntimeError("OpenAI request failed after retries")
    return resp.choices[0].message.content or ""


def openai_chat(*, system: str, user: str, cfg: OpenAIChatConfig) -> str:
    return _openai_chat_complete(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        json_mode=False,
    )


def openai_chat_json(*, system: str, user: str, cfg: OpenAIChatConfig) -> str:
    return _openai_chat_complete(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        json_mode=True,
    )


def compare_pair_openai(
    *,
    ground_truth_avg: list[float],
    pred_a_avg: list[float],
    pred_b_avg: list[float],
    cfg: OpenAIChatConfig,
) -> tuple[str, str]:
    prompt = f"""
Given the ground truth time series data and two model predictions (10-day averages, normalized), determine which prediction aligns better with the ground truth.

Ground Truth: {ground_truth_avg}
Prediction A: {pred_a_avg}
Prediction B: {pred_b_avg}

Respond with exactly one token: A or B.
""".strip()
    raw = openai_chat(
        system="You are a domain expert in time series data. Be strict and answer only with A or B.",
        user=prompt,
        cfg=cfg,
    ).strip()

    choice = raw.split()[0].upper() if raw else ""
    if choice not in {"A", "B"}:
        # Best-effort fallback
        if "A" in raw and "B" not in raw:
            choice = "A"
        elif "B" in raw and "A" not in raw:
            choice = "B"
        else:
            choice = "A"
    return choice, raw


def suggest_weights_openai(
    *,
    desired_relation: str,
    current_weights: DomainMetricWeights,
    score_a: float,
    score_b: float,
    stats: dict[str, Any],
    cfg: OpenAIChatConfig,
) -> tuple[DomainMetricWeights, str]:
    system = "You are an expert in time-series similarity metrics. Output only JSON."

    gt_avg = stats.get("ground_truth_avg")
    a_avg = stats.get("pred_a_avg")
    b_avg = stats.get("pred_b_avg")
    tgt = stats.get("target_scores")

    context_parts: list[str] = []
    if gt_avg is not None and a_avg is not None and b_avg is not None:
        context_parts.append(f"Ground Truth (10-day avg, normalized): {gt_avg}")
        context_parts.append(f"Series A (10-day avg, normalized): {a_avg}")
        context_parts.append(f"Series B (10-day avg, normalized): {b_avg}")
    if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
        context_parts.append(f"Target scores (higher is better): target(A)={tgt[0]}, target(B)={tgt[1]}")
    context = "\n".join(context_parts) if context_parts else "(No time-series context provided.)"

    user = (
        "Adjust the metric weights to satisfy the desired ordering under the current metric.\n\n"
        f"Desired relation: {desired_relation}\n"
        f"Current weights:\n"
        f"- peak_period_weight: {current_weights.peak_period_weight}\n"
        f"- derivative_weight: {current_weights.derivative_weight}\n"
        f"- tolerance_days: {current_weights.tolerance_days}\n"
        f"- amplitude_weight: {current_weights.amplitude_weight}\n\n"
        f"{context}\n\n"
        f"Current scores:\n"
        f"- score(A): {score_a}\n"
        f"- score(B): {score_b}\n\n"
        "Constraints:\n"
        "- peak_period_weight, derivative_weight, amplitude_weight must each be in [0,1] and sum to 1.\n"
        "- tolerance_days must be an integer in [1,30].\n\n"
        "Return ONLY a JSON object with keys:\n"
        '{"peak_period_weight": <0..1>, "derivative_weight": <0..1>, "tolerance_days": <1..30>, "amplitude_weight": <0..1>}\n'
    )

    text = _openai_chat_complete(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        json_mode=True,
    )

    try:
        obj = _extract_weights_object(text)
    except Exception as e:
        raise OpenAIParseError(str(e), raw_output=text) from e
    w = DomainMetricWeights(
        peak_period_weight=_as_float(obj.get("peak_period_weight", current_weights.peak_period_weight)),
        derivative_weight=_as_float(obj.get("derivative_weight", current_weights.derivative_weight)),
        tolerance_days=_as_int(obj.get("tolerance_days", current_weights.tolerance_days)),
        amplitude_weight=_as_float(obj.get("amplitude_weight", current_weights.amplitude_weight)),
    )
    return clamp_weights(w), text
