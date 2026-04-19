from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from rollout.rollout_utils import get_by_candidates, normalize_action


LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
]


@dataclass
class StepResult:
    obs: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class MockSimData:
    def __init__(self) -> None:
        self.ncon = 0
        self.contact = []


class MockSim:
    def __init__(self) -> None:
        self.data = MockSimData()


class MockLiberoEnv:
    """Small deterministic stand-in for smoke tests without MuJoCo/LIBERO."""

    def __init__(self, task_name: str, seed: int = 0, action_dim: int = 7, max_steps: int = 80) -> None:
        self.task_name = task_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.sim = MockSim()
        self.step_count = 0
        self.eef_pos = np.array([0.35, 0.0, 0.25], dtype=np.float32)
        self.object_pos = np.array([0.48, 0.0, 0.03], dtype=np.float32)
        self.basket_pos = np.array([0.18, 0.20, 0.03], dtype=np.float32)
        self.gripper = np.array([0.04], dtype=np.float32)
        self.object_grasped = False

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.sim.data.ncon = 0
        self.eef_pos = np.array([0.35, 0.0, 0.25], dtype=np.float32)
        jitter = self.rng.normal(0.0, 0.01, size=2).astype(np.float32)
        self.object_pos = np.array([0.48 + jitter[0], jitter[1], 0.03], dtype=np.float32)
        self.gripper = np.array([0.04], dtype=np.float32)
        self.object_grasped = False
        return self._obs()

    def step(self, action: Sequence[float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action = normalize_action(action, self.action_dim)
        self.step_count += 1
        self.eef_pos = self.eef_pos + np.clip(action[:3], -0.03, 0.03)
        if action.size:
            self.gripper[0] = np.clip(self.gripper[0] + 0.01 * np.sign(action[-1]), 0.0, 0.08)
        dist = float(np.linalg.norm(self.eef_pos - self.object_pos))
        self.sim.data.ncon = 1 if dist < 0.12 else 0
        if self.sim.data.ncon and self.gripper[0] < 0.035:
            self.object_grasped = True
        if self.object_grasped:
            self.object_pos = self.eef_pos + np.array([0.0, 0.0, -0.045], dtype=np.float32)
        success = float(np.linalg.norm(self.object_pos[:2] - self.basket_pos[:2])) < 0.08 and self.object_pos[2] < 0.08
        reward = 1.0 if success else 0.0
        done = bool(success or self.step_count >= self.max_steps)
        return self._obs(), reward, done, {"success": bool(success), "step": self.step_count}

    def render(self) -> np.ndarray:
        return self._image()

    def close(self) -> None:
        return None

    def _image(self) -> np.ndarray:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[:, :, 1] = 30
        obj = np.clip((self.object_pos[:2] + np.array([0.1, 0.3])) * 130, 5, 90).astype(int)
        eef = np.clip((self.eef_pos[:2] + np.array([0.1, 0.3])) * 130, 5, 90).astype(int)
        image[obj[1] - 3 : obj[1] + 4, obj[0] - 3 : obj[0] + 4] = np.array([220, 40, 40], dtype=np.uint8)
        image[eef[1] - 2 : eef[1] + 3, eef[0] - 2 : eef[0] + 3] = np.array([40, 140, 240], dtype=np.uint8)
        return image

    def _obs(self) -> Dict[str, Any]:
        return {
            "agentview_image": self._image(),
            "robot0_eef_pos": self.eef_pos.copy(),
            "robot0_gripper_qpos": self.gripper.copy(),
            "target_object_pos": self.object_pos.copy(),
            "basket_pos": self.basket_pos.copy(),
            "task_name": self.task_name,
        }


class LiberoEnvAdapter:
    def __init__(self, env: Any, cfg: Mapping[str, Any], task_name: str) -> None:
        self.env = env
        self.cfg = cfg
        self.task_name = task_name
        self.sim = getattr(env, "sim", None)

    def reset(self) -> Dict[str, Any]:
        obs = self.env.reset()
        return dict(obs or {})

    def step(self, action: Sequence[float]) -> StepResult:
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
        return StepResult(dict(obs or {}), float(reward), bool(done), dict(info or {}))

    def render(self) -> Optional[np.ndarray]:
        if hasattr(self.env, "render"):
            frame = self.env.render()
            return np.asarray(frame) if frame is not None else None
        return None

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def get_libero_object_tasks() -> Sequence[str]:
    return list(LIBERO_OBJECT_TASKS)


def _load_factory(path: str):
    module_name, function_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def create_env(task_name: str, cfg: Mapping[str, Any], seed: int = 0) -> LiberoEnvAdapter:
    env_cfg = cfg.get("env", {})
    policy_cfg = cfg.get("policy", {})
    if env_cfg.get("mode", "mock") == "mock":
        env = MockLiberoEnv(
            task_name=task_name,
            seed=seed,
            action_dim=int(policy_cfg.get("action_dim", 7)),
            max_steps=int(cfg.get("max_steps", 220)),
        )
        return LiberoEnvAdapter(env, cfg, task_name)

    factory_path = env_cfg.get("env_factory")
    if factory_path:
        factory = _load_factory(factory_path)
        env = factory(task_name=task_name, seed=seed, config=cfg)
        return LiberoEnvAdapter(env, cfg, task_name)

    raise RuntimeError(
        "Real LIBERO environment creation is version-specific. Set env.env_factory "
        "to 'module:function' in the config, or use --mock for smoke tests. "
        "The baseline script still calls the official OpenVLA-OFT run_libero_eval.py."
    )


def extract_image(obs: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[np.ndarray]:
    value = get_by_candidates(obs, cfg.get("env", {}).get("image_key_candidates", []))
    return None if value is None else np.asarray(value)


def extract_eef_pos(obs: Mapping[str, Any], cfg: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        get_by_candidates(obs, cfg.get("env", {}).get("eef_key_candidates", []), np.zeros(3)),
        dtype=np.float32,
    ).reshape(-1)[:3]


def extract_object_pos(obs: Mapping[str, Any], cfg: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        get_by_candidates(obs, cfg.get("env", {}).get("object_key_candidates", []), np.zeros(3)),
        dtype=np.float32,
    ).reshape(-1)[:3]


def extract_gripper(obs: Mapping[str, Any], cfg: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        get_by_candidates(obs, cfg.get("env", {}).get("gripper_key_candidates", []), np.zeros(1)),
        dtype=np.float32,
    ).reshape(-1)


def check_success(info: Mapping[str, Any], reward: float = 0.0) -> bool:
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return bool(info[key])
    return bool(reward > 0.5)
