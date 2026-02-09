from __future__ import annotations

from typing import Literal

from .metric import DomainMetricWeights


SceneKey = Literal["peak_period", "slope_curvature", "amplitude"]


SCENE_TARGET_WEIGHTS: dict[SceneKey, DomainMetricWeights] = {
    # Mirrors `domain_metric/src/policy_concrete.py` target weights.
    "peak_period": DomainMetricWeights(
        peak_period_weight=0.8,
        derivative_weight=0.1,
        tolerance_days=5,
        amplitude_weight=0.1,
    ),
    "slope_curvature": DomainMetricWeights(
        peak_period_weight=0.1,
        derivative_weight=0.8,
        tolerance_days=5,
        amplitude_weight=0.1,
    ),
    "amplitude": DomainMetricWeights(
        peak_period_weight=0.1,
        derivative_weight=0.1,
        tolerance_days=5,
        amplitude_weight=0.8,
    ),
}


SCENE_ALIASES: dict[str, SceneKey] = {
    # Canonical
    "peak_period": "peak_period",
    "slope_curvature": "slope_curvature",
    "amplitude": "amplitude",
    # Legacy names (from `domain_metric/src/policy_concrete.py`)
    "peak_period_weight": "peak_period",
    "derivative_weight": "slope_curvature",
    "amplitude_weight": "amplitude",
}
