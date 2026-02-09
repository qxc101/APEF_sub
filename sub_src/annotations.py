from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re


@dataclass(frozen=True)
class PairwiseComparison:
    ts_a: int
    ts_b: int
    # 1 means A wins, 2 means B wins, 0 means tie/unknown
    winner: int


_TS_RE = re.compile(r"TS\s*0*([0-9]+)\b", re.IGNORECASE)


def parse_ts_id(value: str) -> int:
    m = _TS_RE.search(str(value).strip())
    if not m:
        raise ValueError(f"Could not parse TS id from: {value!r}")
    return int(m.group(1))


def _pick_first_present(row: dict[str, str], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return None


def read_pairwise_annotations_csv(
    path: Path,
    *,
    winner_column: str,
    model1_columns: list[str],
    model2_columns: list[str],
) -> list[PairwiseComparison]:
    comps: list[PairwiseComparison] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m1 = _pick_first_present(row, model1_columns)
            m2 = _pick_first_present(row, model2_columns)
            if m1 is None or m2 is None:
                continue
            winner_raw = str(row.get(winner_column, "")).strip()
            if winner_raw == "":
                continue
            try:
                winner = int(float(winner_raw))
            except ValueError:
                continue
            if winner not in {0, 1, 2}:
                continue
            comps.append(PairwiseComparison(ts_a=parse_ts_id(m1), ts_b=parse_ts_id(m2), winner=winner))
    return comps


def win_loss_scores(comparisons: list[PairwiseComparison], *, n_models: int = 20) -> list[int]:
    scores = [0 for _ in range(n_models)]
    for c in comparisons:
        if not (0 <= c.ts_a < n_models and 0 <= c.ts_b < n_models):
            continue
        if c.winner == 1:
            scores[c.ts_a] += 1
            scores[c.ts_b] -= 1
        elif c.winner == 2:
            scores[c.ts_a] -= 1
            scores[c.ts_b] += 1
    return scores

