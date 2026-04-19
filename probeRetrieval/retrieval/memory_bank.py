from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from rollout.rollout_utils import ensure_dir, json_default


@dataclass
class MemoryItem:
    episode_id: str
    task_name: str
    image_embedding: np.ndarray
    raw_image_path: str | None
    action_v_t0: np.ndarray
    probe_features: Dict[str, float]
    post_probe_action_chunk: np.ndarray | None
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self, array_index: int) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("image_embedding")
        data.pop("action_v_t0")
        data.pop("post_probe_action_chunk")
        data["array_index"] = array_index
        return data


class MemoryBank:
    def __init__(self, items: Sequence[MemoryItem] | None = None) -> None:
        self.items: List[MemoryItem] = list(items or [])

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def save(self, directory: str | Path) -> None:
        directory = ensure_dir(directory)
        metadata_path = directory / "metadata.jsonl"
        arrays_path = directory / "arrays.npz"
        embeddings = np.stack([item.image_embedding for item in self.items], axis=0) if self.items else np.zeros((0, 0))
        actions = np.stack([item.action_v_t0 for item in self.items], axis=0) if self.items else np.zeros((0, 0))
        chunks = [item.post_probe_action_chunk for item in self.items]
        chunk_obj = np.empty(len(chunks), dtype=object)
        for idx, chunk in enumerate(chunks):
            chunk_obj[idx] = chunk
        np.savez_compressed(arrays_path, image_embeddings=embeddings, action_v_t0=actions, post_probe_action_chunks=chunk_obj)
        with metadata_path.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(self.items):
                f.write(json.dumps(item.to_metadata(idx), default=json_default, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryBank":
        directory = Path(directory)
        metadata_path = directory / "metadata.jsonl"
        arrays_path = directory / "arrays.npz"
        if not metadata_path.exists() or not arrays_path.exists():
            raise FileNotFoundError(f"Memory bank not found in {directory}")
        arrays = np.load(arrays_path, allow_pickle=True)
        metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        items: List[MemoryItem] = []
        for row in metadata:
            idx = int(row.pop("array_index"))
            items.append(
                MemoryItem(
                    episode_id=row["episode_id"],
                    task_name=row["task_name"],
                    image_embedding=np.asarray(arrays["image_embeddings"][idx], dtype=np.float32),
                    raw_image_path=row.get("raw_image_path"),
                    action_v_t0=np.asarray(arrays["action_v_t0"][idx], dtype=np.float32),
                    probe_features={k: float(v) for k, v in row.get("probe_features", {}).items()},
                    post_probe_action_chunk=arrays["post_probe_action_chunks"][idx],
                    success=bool(row.get("success", False)),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return cls(items)

    def image_matrix(self) -> np.ndarray:
        return np.stack([item.image_embedding for item in self.items], axis=0) if self.items else np.zeros((0, 0), dtype=np.float32)

    def action_matrix(self) -> np.ndarray:
        return np.stack([item.action_v_t0 for item in self.items], axis=0) if self.items else np.zeros((0, 0), dtype=np.float32)

    def ids(self) -> List[str]:
        return [item.episode_id for item in self.items]

    def successful_indices(self, indices: Iterable[int]) -> List[int]:
        return [idx for idx in indices if self.items[idx].success]
