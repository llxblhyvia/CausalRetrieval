from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np

from rollout.rollout_utils import ensure_dir


def normalize_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: int = 20) -> Path | None:
    frames = [normalize_frame(frame) for frame in frames]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return None
    path = Path(path)
    ensure_dir(path.parent)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


class FrameBuffer:
    def __init__(self, maxlen: int) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, frame: np.ndarray | None) -> None:
        normalized = normalize_frame(frame)
        if normalized is not None:
            self._frames.append(normalized)

    def to_list(self) -> list[np.ndarray]:
        return list(self._frames)


def should_save_video(episode_idx: int, every: int) -> bool:
    return every > 0 and episode_idx % every == 0


def extend_frames(dest: list[np.ndarray], frames: Sequence[np.ndarray | None]) -> None:
    for frame in frames:
        normalized = normalize_frame(frame)
        if normalized is not None:
            dest.append(normalized)
