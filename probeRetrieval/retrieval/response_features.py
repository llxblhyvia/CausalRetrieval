from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


RESPONSE_FEATURE_KEYS = [
    "probe_start_step",
    "contact_steps",
    "contact_ratio",
    "mean_force",
    "max_force",
    "force_std",
    "probe_obj_displacement",
    "end_effector_movement",
    "probe_motion_ratio",
    "post_probe_object_to_target_distance",
]

DEFAULT_RESPONSE_WEIGHTS = {
    "probe_start_step": 0.25,
    "contact_steps": 0.75,
    "contact_ratio": 1.0,
    "mean_force": 1.25,
    "max_force": 1.0,
    "force_std": 0.5,
    "probe_obj_displacement": 1.5,
    "end_effector_movement": 0.5,
    "probe_motion_ratio": 1.5,
    "post_probe_object_to_target_distance": 1.0,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isnan(value):
                return float(default)
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def probe_motion_ratio(features: Mapping[str, Any], eps: float = 1.0e-6) -> float:
    displacement = safe_float(features.get("probe_obj_displacement"), 0.0)
    ee_motion = safe_float(features.get("end_effector_movement"), 0.0)
    return float(displacement / (ee_motion + eps))


def enrich_response_features(features: Mapping[str, Any], eps: float = 1.0e-6) -> dict[str, float]:
    out = {str(key): safe_float(value) for key, value in dict(features).items()}
    out["probe_motion_ratio"] = probe_motion_ratio(out, eps=eps)
    return out


def response_feature_vector(
    features: Mapping[str, Any],
    keys: Sequence[str] | None = None,
    eps: float = 1.0e-6,
) -> np.ndarray:
    enriched = enrich_response_features(features, eps=eps)
    keys = list(keys or RESPONSE_FEATURE_KEYS)
    return np.asarray([safe_float(enriched.get(key), 0.0) for key in keys], dtype=np.float32)


def response_weight_vector(keys: Sequence[str] | None = None, custom: Mapping[str, float] | None = None) -> np.ndarray:
    keys = list(keys or RESPONSE_FEATURE_KEYS)
    merged = dict(DEFAULT_RESPONSE_WEIGHTS)
    if custom:
        merged.update({str(k): float(v) for k, v in custom.items()})
    return np.asarray([float(merged.get(key, 1.0)) for key in keys], dtype=np.float32)


def weighted_z_l1_distance(
    query_features: Mapping[str, Any],
    candidate_features: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    keys: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    eps: float = 1.0e-6,
) -> float:
    query_vec = response_feature_vector(query_features, keys=keys, eps=eps)
    candidate_vec = response_feature_vector(candidate_features, keys=keys, eps=eps)
    weight_vec = response_weight_vector(keys=keys, custom=weights)
    q_norm = (query_vec - mean) / (std + eps)
    c_norm = (candidate_vec - mean) / (std + eps)
    return float(np.abs(q_norm - c_norm).dot(weight_vec))

