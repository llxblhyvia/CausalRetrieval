from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from probe.feature_extractor import FEATURE_KEYS, feature_vector
from retrieval.memory_bank import MemoryBank


def rerank_by_probe(
    query_features: Mapping[str, float],
    candidates: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    top_k: int,
    image_weight: float = 0.35,
    probe_weight: float = 0.65,
    eps: float = 1.0e-6,
) -> List[Dict[str, float]]:
    if not candidates:
        return []
    query_vec = feature_vector(query_features, FEATURE_KEYS)
    candidate_vecs = np.stack([feature_vector(bank.items[int(c["index"])].probe_features, FEATURE_KEYS) for c in candidates], axis=0)
    mean = candidate_vecs.mean(axis=0, keepdims=True)
    std = candidate_vecs.std(axis=0, keepdims=True)
    c_norm = (candidate_vecs - mean) / (std + eps)
    q_norm = (query_vec.reshape(1, -1) - mean) / (std + eps)
    distances = np.linalg.norm(c_norm - q_norm, axis=1)
    probe_scores = 1.0 / (1.0 + distances)
    image_scores = np.asarray([float(c.get("score", 0.0)) for c in candidates], dtype=np.float32)
    if image_scores.size and image_scores.max() > image_scores.min():
        image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min() + eps)
    combined = float(image_weight) * image_scores + float(probe_weight) * probe_scores
    order = np.argsort(-combined)[: int(top_k)]
    out: List[Dict[str, float]] = []
    for local_idx in order:
        c = dict(candidates[int(local_idx)])
        c["probe_score"] = float(probe_scores[int(local_idx)])
        c["combined_score"] = float(combined[int(local_idx)])
        out.append(c)
    return out


def aggregate_retrieved_action(
    ranked: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    successful_only: bool = True,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray | None, List[str]]:
    selected = []
    weights = []
    ids = []
    for candidate in ranked:
        idx = int(candidate["index"])
        item = bank.items[idx]
        if successful_only and not item.success:
            continue
        if item.post_probe_action_chunk is not None and np.asarray(item.post_probe_action_chunk).size:
            selected.append(np.asarray(item.post_probe_action_chunk, dtype=np.float32).reshape(-1, item.action_v_t0.size)[0])
        else:
            selected.append(item.action_v_t0.astype(np.float32))
        weights.append(float(candidate.get("combined_score", candidate.get("score", 1.0))))
        ids.append(item.episode_id)
    if not selected:
        return None, ids
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr = np.maximum(weight_arr, 0.0)
    if float(weight_arr.sum()) <= eps:
        weight_arr = np.ones_like(weight_arr)
    weight_arr = weight_arr / (weight_arr.sum() + eps)
    actions = np.stack(selected, axis=0)
    return (actions * weight_arr[:, None]).sum(axis=0).astype(np.float32), ids
