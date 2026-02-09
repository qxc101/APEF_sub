from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # `.../domain_metric/sub_src/paths.py` -> `.../domain_metric`
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def results_dir() -> Path:
    return project_root() / "sub_results"


def ensure_results_dir() -> Path:
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out

