from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SimpleImageEmbedder:
    """Dependency-light image embedding for stable first-stage retrieval."""

    def __init__(self, bins: int = 16, grid: int = 8) -> None:
        self.bins = bins
        self.grid = grid

    def embed(self, image: np.ndarray | None) -> np.ndarray:
        if image is None:
            return np.zeros(self.bins * 3 + self.grid * self.grid * 3, dtype=np.float32)
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        arr = arr[:, :, :3].astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=self.bins, range=(0, 255), density=True)
            hist_parts.append(hist.astype(np.float32))
        small = self._resize_mean(arr, self.grid, self.grid).reshape(-1).astype(np.float32) / 255.0
        emb = np.concatenate(hist_parts + [small], axis=0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @staticmethod
    def _resize_mean(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        h, w = arr.shape[:2]
        y_edges = np.linspace(0, h, out_h + 1, dtype=int)
        x_edges = np.linspace(0, w, out_w + 1, dtype=int)
        out = np.zeros((out_h, out_w, arr.shape[2]), dtype=np.float32)
        for y in range(out_h):
            for x in range(out_w):
                patch = arr[y_edges[y] : max(y_edges[y + 1], y_edges[y] + 1), x_edges[x] : max(x_edges[x + 1], x_edges[x] + 1)]
                out[y, x] = patch.mean(axis=(0, 1))
        return out


def create_image_embedder(cfg: Mapping[str, Any]):
    backend = cfg.get("retrieval", {}).get("image_backend", "simple")
    if backend != "simple":
        raise RuntimeError(
            f"Unsupported image_backend {backend!r}. The bundled stable backend is 'simple'; "
            "add a CLIP factory once torch/transformers are installed."
        )
    return SimpleImageEmbedder()
