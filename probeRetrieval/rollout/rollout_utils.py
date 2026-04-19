from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config files. Install it in "
            "/network/rit/lab/wang_lab_cs/yhan/envs before running this project."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = CONFIG_DIR / config_path
    data = _load_yaml(config_path)
    parent_name = data.pop("inherits", None)
    if parent_name:
        parent = load_config(parent_name)
        return dict(deep_update(parent, data))
    return data


def parse_unknown_overrides(items: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items:
        if not item.startswith("--") or "=" not in item:
            raise ValueError(f"Overrides must look like --a.b=value, got {item!r}")
        key, raw_value = item[2:].split("=", 1)
        value: Any = raw_value
        if raw_value.lower() in {"true", "false"}:
            value = raw_value.lower() == "true"
        else:
            for caster in (int, float):
                try:
                    value = caster(raw_value)
                    break
                except ValueError:
                    pass
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return overrides


def add_common_args(parser: argparse.ArgumentParser, default_config: str = "default.yaml") -> None:
    parser.add_argument("--config", default=default_config, help="YAML config path or name in configs/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_trials_per_task", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--memory_dir", default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock env and policy")


def config_from_args(args: argparse.Namespace, unknown: Sequence[str] = ()) -> Dict[str, Any]:
    cfg = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "num_trials_per_task", None) is not None:
        cfg["num_trials_per_task"] = args.num_trials_per_task
    if getattr(args, "output_dir", None):
        cfg.setdefault("paths", {})["output_dir"] = args.output_dir
    if getattr(args, "memory_dir", None):
        cfg.setdefault("paths", {})["memory_dir"] = args.memory_dir
    if getattr(args, "mock", False):
        cfg.setdefault("env", {})["mode"] = "mock"
        cfg.setdefault("policy", {})["mode"] = "mock"
    if unknown:
        deep_update(cfg, parse_unknown_overrides(unknown))
    return cfg


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=json_default, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_by_candidates(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def to_numpy(value: Any, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def normalize_action(action: Any, action_dim: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size < action_dim:
        arr = np.pad(arr, (0, action_dim - arr.size))
    elif arr.size > action_dim:
        arr = arr[:action_dim]
    return arr


def copy_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return deepcopy(dict(cfg))
