from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from retrieval.memory_bank import MemoryBank
from retrieval.response_bank import ResponseBank
from retrieval.response_features import RESPONSE_FEATURE_KEYS, weighted_z_l1_distance


def filter_candidates_by_task_and_probe(
    candidates: Sequence[Mapping[str, float]],
    response_bank: ResponseBank,
    task_name: str,
) -> List[Dict[str, float]]:
    filtered: List[Dict[str, float]] = []
    for candidate in candidates:
        episode_id = str(candidate.get("episode_id", ""))
        item = response_bank.get(episode_id)
        if item is None or item.task_name != task_name or not item.probe_triggered:
            continue
        out = dict(candidate)
        out["response_episode_id"] = episode_id
        out["response_success"] = bool(item.final_success)
        filtered.append(out)
    return filtered


def rerank_by_response(
    query_features: Mapping[str, float],
    candidates: Sequence[Mapping[str, float]],
    response_bank: ResponseBank,
    task_name: str,
    top_k: int,
    feature_keys: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | None = None,
    eps: float = 1.0e-6,
) -> List[Dict[str, float]]:
    if not candidates:
        return []
    feature_keys = list(feature_keys or RESPONSE_FEATURE_KEYS)
    filtered = filter_candidates_by_task_and_probe(candidates, response_bank, task_name)
    if not filtered:
        return []
    mean, std = response_bank.task_feature_stats(task_name, probe_triggered_only=True, keys=feature_keys, eps=eps)
    distances = []
    for candidate in filtered:
        item = response_bank.get(str(candidate["episode_id"]))
        if item is None:
            distances.append(np.inf)
            continue
        distances.append(
            weighted_z_l1_distance(
                query_features,
                item.response_features,
                mean,
                std,
                keys=feature_keys,
                weights=feature_weights,
                eps=eps,
            )
        )
    order = np.argsort(np.asarray(distances, dtype=np.float32))[: int(top_k)]
    out: List[Dict[str, float]] = []
    for local_idx in order:
        c = dict(filtered[int(local_idx)])
        c["response_distance"] = float(distances[int(local_idx)])
        out.append(c)
    return out


def select_successful_from_topk(
    ranked: Sequence[Mapping[str, float]],
) -> List[Dict[str, float]]:
    return [dict(candidate) for candidate in ranked if bool(candidate.get("response_success", candidate.get("success", False)))]


def aggregate_actions_soft(
    ranked: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    response_bank: ResponseBank | None = None,
    successful_only: bool = True,
    temperature: float = 0.1,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray | None, List[str]]:
    selected = []
    weights = []
    ids = []
    for candidate in ranked:
        idx = int(candidate["index"])
        item = bank.items[idx]
        is_success = bool(item.success)
        if response_bank is not None:
            response_item = response_bank.get(item.episode_id)
            if response_item is not None:
                is_success = bool(response_item.final_success)
        if successful_only and not is_success:
            continue
        if item.post_probe_action_chunk is not None and np.asarray(item.post_probe_action_chunk).size:
            selected.append(np.asarray(item.post_probe_action_chunk, dtype=np.float32).reshape(-1, item.action_v_t0.size)[0])
        else:
            selected.append(item.action_v_t0.astype(np.float32))
        distance = float(candidate.get("response_distance", 0.0))
        weights.append(float(np.exp(-distance / max(float(temperature), eps))))
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


def aggregate_actions_major_vote(
    ranked: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    response_bank: ResponseBank | None = None,
    successful_only: bool = True,
    decimals: int = 2,
) -> tuple[np.ndarray | None, List[str]]:
    votes: dict[tuple[float, ...], int] = {}
    exemplars: dict[tuple[float, ...], np.ndarray] = {}
    ids: list[str] = []
    for candidate in ranked:
        idx = int(candidate["index"])
        item = bank.items[idx]
        is_success = bool(item.success)
        if response_bank is not None:
            response_item = response_bank.get(item.episode_id)
            if response_item is not None:
                is_success = bool(response_item.final_success)
        if successful_only and not is_success:
            continue
        if item.post_probe_action_chunk is not None and np.asarray(item.post_probe_action_chunk).size:
            action = np.asarray(item.post_probe_action_chunk, dtype=np.float32).reshape(-1, item.action_v_t0.size)[0]
        else:
            action = item.action_v_t0.astype(np.float32)
        key = tuple(np.round(action.astype(np.float32), decimals=decimals).tolist())
        votes[key] = votes.get(key, 0) + 1
        exemplars[key] = action
        ids.append(item.episode_id)
    if not votes:
        return None, ids
    best_key = max(votes.items(), key=lambda kv: kv[1])[0]
    return exemplars[best_key].astype(np.float32), ids


def rerank_by_probe(
    query_features: Mapping[str, float],
    candidates: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    top_k: int,
    image_weight: float = 0.35,
    probe_weight: float = 0.65,
    eps: float = 1.0e-6,
) -> List[Dict[str, float]]:
    del bank, image_weight, probe_weight, eps
    out = [dict(candidate) for candidate in candidates[: int(top_k)]]
    for candidate in out:
        candidate["response_distance"] = float(candidate.get("response_distance", 0.0))
    return out


def aggregate_retrieved_action(
    ranked: Sequence[Mapping[str, float]],
    bank: MemoryBank,
    successful_only: bool = True,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray | None, List[str]]:
    return aggregate_actions_soft(
        ranked,
        bank,
        response_bank=None,
        successful_only=successful_only,
        temperature=0.1,
        eps=eps,
    )
