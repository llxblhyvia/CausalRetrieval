from __future__ import annotations

from typing import Dict, List

import numpy as np

from retrieval.memory_bank import MemoryBank


def cosine_scores(query: np.ndarray, matrix: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    matrix = np.asarray(matrix, dtype=np.float32)
    q_norm = np.linalg.norm(query, axis=1, keepdims=True)
    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return ((matrix @ query.T).reshape(-1) / ((m_norm.reshape(-1) * q_norm.reshape(-1)[0]) + eps)).astype(np.float32)


def retrieve_top_k(query_embedding: np.ndarray, bank: MemoryBank, k: int, eps: float = 1.0e-6) -> List[Dict[str, float]]:
    scores = cosine_scores(query_embedding, bank.image_matrix(), eps=eps)
    if scores.size == 0:
        return []
    order = np.argsort(-scores)[: int(k)]
    return [{"index": int(idx), "score": float(scores[idx]), "episode_id": bank.items[int(idx)].episode_id} for idx in order]
