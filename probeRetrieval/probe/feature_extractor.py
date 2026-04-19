from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from probe.probe_runner import ProbeTrace, trace_state_vectors


FEATURE_KEYS = [
    "contact_ratio",
    "total_ee_motion",
    "total_obj_motion",
    "motion_ratio",
    "gripper_delta",
]


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def extract_probe_features(trace: ProbeTrace, cfg: Mapping[str, Any]) -> Dict[str, float]:
    eps = float(cfg.get("retrieval", {}).get("eps", 1.0e-6))
    states = trace_state_vectors(trace, cfg)
    total_ee_motion = path_length(states["eef_pos"])
    total_obj_motion = path_length(states["object_pos"])
    contacts = states["contact_counts"]
    denominator = max(len(contacts), 1)
    gripper = states["gripper"]
    gripper_start = float(gripper[0].mean()) if gripper.size else 0.0
    gripper_end = float(gripper[-1].mean()) if gripper.size else 0.0
    return {
        "contact_ratio": float((contacts > 0).sum() / denominator),
        "total_ee_motion": total_ee_motion,
        "total_obj_motion": total_obj_motion,
        "motion_ratio": float(total_obj_motion / (total_ee_motion + eps)),
        "gripper_delta": float(gripper_end - gripper_start),
    }


def feature_vector(features: Mapping[str, Any], keys: list[str] | None = None) -> np.ndarray:
    keys = keys or FEATURE_KEYS
    return np.asarray([float(features.get(key, 0.0)) for key in keys], dtype=np.float32)


def normalize_feature_matrix(matrix: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    return ((matrix - mean) / (std + eps)).astype(np.float32)
