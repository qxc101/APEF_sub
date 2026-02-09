from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .paths import data_dir


DatasetKey = Literal["co2", "gpp", "camels", "xmethanewet"]


@dataclass(frozen=True)
class DatasetRun:
    key: DatasetKey
    feature_name: str
    ground_truth: np.ndarray  # (365,)
    predictions: np.ndarray  # (20, 365)
    ts_labels: list[str]  # ["TS 0", ..., "TS 19"]


def _load_predictions(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] != 1:
        raise ValueError(f"Expected predictions shape (n, 365, 1), got {arr.shape} from {path}")
    return arr[:, :, 0].astype(float)


def load_dataset_run(dataset: DatasetKey) -> DatasetRun:
    root = data_dir()
    ts_labels = [f"TS {i}" for i in range(20)]

    if dataset in {"co2", "gpp"}:
        gt_path = root / "GT" / "Y_real_scaled_ECfluxnet_Combined_0.npy"
        gt_all = np.load(gt_path)
        year_start = 365 * 5
        year_end = 365 * 6
        site_idx = 0
        if dataset == "co2":
            var_idx = 1
            preds_path = root / "sampled_combined_CO2_0_5.npy"
            feature_name = "CO2"
        else:
            var_idx = 0
            preds_path = root / "sampled_combined_GPP_0_5.npy"
            feature_name = "GPP"

        gt = gt_all[year_start:year_end, site_idx, var_idx].reshape(-1).astype(float)
        preds = _load_predictions(preds_path)
        if gt.shape[0] != preds.shape[1]:
            raise ValueError(f"GT length {gt.shape[0]} != preds length {preds.shape[1]}")
        return DatasetRun(
            key=dataset,
            feature_name=feature_name,
            ground_truth=gt,
            predictions=preds,
            ts_labels=ts_labels,
        )

    exp2_dir = root / "exp2_camels_methane_annotations"
    if dataset == "camels":
        gt = np.load(exp2_dir / "gt_camels_01013500_1980.npy").reshape(-1).astype(float)
        preds = _load_predictions(exp2_dir / "sampled_combined_camels_01013500_1980.npy")
        return DatasetRun(
            key=dataset,
            feature_name="discharge",
            ground_truth=gt,
            predictions=preds,
            ts_labels=ts_labels,
        )

    if dataset == "xmethanewet":
        gt = np.load(exp2_dir / "gt_xmethanewet_AT.Neu_2010_FCH4_F_ANNOPTLM.npy").reshape(-1).astype(float)
        preds = _load_predictions(
            exp2_dir / "sampled_combined_xmethanewet_AT.Neu_2010_FCH4_F_ANNOPTLM.npy"
        )
        return DatasetRun(
            key=dataset,
            feature_name="CH4",
            ground_truth=gt,
            predictions=preds,
            ts_labels=ts_labels,
        )

    raise ValueError(f"Unknown dataset: {dataset}")

