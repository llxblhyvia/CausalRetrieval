from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from retrieval.response_features import enrich_response_features


def make_episode_id(task_id: int | None, episode_idx: int | None) -> str | None:
    if task_id is None or episode_idx is None:
        return None
    return f"task_{int(task_id):02d}/trial_{int(episode_idx):04d}"


@dataclass
class ResponseItem:
    episode_id: str
    task_name: str
    task_id: int | None
    episode_idx: int | None
    probe_triggered: bool
    final_success: bool
    response_features: Dict[str, float]
    metadata: Dict[str, Any]


class ResponseBank:
    def __init__(self, items: Sequence[ResponseItem] | None = None) -> None:
        self.items: List[ResponseItem] = list(items or [])
        self._by_episode_id = {item.episode_id: item for item in self.items}

    def __len__(self) -> int:
        return len(self.items)

    def get(self, episode_id: str) -> ResponseItem | None:
        return self._by_episode_id.get(str(episode_id))

    def lookup_candidates(self, candidates: Iterable[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], ResponseItem]]:
        out: list[tuple[Mapping[str, Any], ResponseItem]] = []
        for candidate in candidates:
            item = self.get(str(candidate.get("episode_id", "")))
            if item is not None:
                out.append((candidate, item))
        return out

    def task_items(self, task_name: str, *, probe_triggered_only: bool = False) -> list[ResponseItem]:
        rows = [item for item in self.items if item.task_name == task_name]
        if probe_triggered_only:
            rows = [item for item in rows if item.probe_triggered]
        return rows

    def task_feature_stats(
        self,
        task_name: str,
        *,
        probe_triggered_only: bool = True,
        keys: Sequence[str] | None = None,
        eps: float = 1.0e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        from retrieval.response_features import RESPONSE_FEATURE_KEYS, response_feature_vector

        keys = list(keys or RESPONSE_FEATURE_KEYS)
        rows = self.task_items(task_name, probe_triggered_only=probe_triggered_only)
        if not rows:
            return np.zeros((len(keys),), dtype=np.float32), np.ones((len(keys),), dtype=np.float32)
        matrix = np.stack([response_feature_vector(item.response_features, keys=keys, eps=eps) for item in rows], axis=0)
        mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return mean, std

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ResponseBank":
        path = Path(path)
        items: list[ResponseItem] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            episode_idx = row.get("episode_idx")
            episode_id = row.get("episode_id") or make_episode_id(task_id, episode_idx)
            if not episode_id:
                continue
            response_features = enrich_response_features(
                {
                    "probe_start_step": row.get("probe_start_step"),
                    "contact_steps": row.get("contact_steps"),
                    "contact_ratio": row.get("contact_ratio"),
                    "mean_force": row.get("mean_force"),
                    "max_force": row.get("max_force"),
                    "force_std": row.get("force_std"),
                    "probe_obj_displacement": row.get("probe_obj_displacement"),
                    "end_effector_movement": row.get("end_effector_movement"),
                    "post_probe_object_to_target_distance": row.get("post_probe_object_to_target_distance"),
                }
            )
            metadata = {
                "setting_name": row.get("setting_name"),
                "sweep_type": row.get("sweep_type"),
                "friction_value": row.get("friction_value"),
                "mass_scale": row.get("mass_scale"),
                "target_instance": row.get("target_instance"),
                "target_reference": row.get("target_reference"),
                "final_object_to_target_distance": row.get("final_object_to_target_distance"),
                "video_path": row.get("video_path"),
                "probe_video_path": row.get("probe_video_path"),
            }
            items.append(
                ResponseItem(
                    episode_id=str(episode_id),
                    task_name=str(row.get("task_name", "")),
                    task_id=int(task_id) if task_id is not None else None,
                    episode_idx=int(episode_idx) if episode_idx is not None else None,
                    probe_triggered=bool(row.get("probe_triggered", False)),
                    final_success=bool(row.get("final_success", False)),
                    response_features=response_features,
                    metadata=metadata,
                )
            )
        return cls(items)
