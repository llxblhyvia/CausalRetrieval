from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from env.libero_wrapper import extract_eef_pos, extract_gripper, extract_object_pos
from rollout.rollout_utils import normalize_action


@dataclass
class ProbeTrace:
    observations: List[Dict[str, Any]]
    actions: List[np.ndarray]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    contact_counts: List[int]
    done: bool = False


class ProbeRunner:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.action_dim = int(cfg.get("policy", {}).get("action_dim", 7))
        probe_cfg = cfg.get("probe", {})
        self.num_close_steps = int(probe_cfg.get("num_close_steps", 2))
        self.num_lift_steps = int(probe_cfg.get("num_lift_steps", 3))
        self.num_hold_steps = int(probe_cfg.get("num_hold_steps", 1))
        self.close_delta = float(probe_cfg.get("close_delta", -0.15))
        self.lift_delta_z = float(probe_cfg.get("lift_delta_z", 0.015))
        self.max_action_abs = float(probe_cfg.get("max_action_abs", 1.0))

    def make_sequence(self, action_v_t0: Sequence[float] | None = None) -> List[np.ndarray]:
        sequence: List[np.ndarray] = []
        base = np.zeros(self.action_dim, dtype=np.float32)
        if action_v_t0 is not None:
            base = normalize_action(action_v_t0, self.action_dim) * 0.0
        for _ in range(self.num_close_steps):
            action = base.copy()
            action[-1] = self.close_delta
            sequence.append(np.clip(action, -self.max_action_abs, self.max_action_abs))
        for _ in range(self.num_lift_steps):
            action = base.copy()
            action[2] = self.lift_delta_z
            action[-1] = self.close_delta
            sequence.append(np.clip(action, -self.max_action_abs, self.max_action_abs))
        for _ in range(self.num_hold_steps):
            action = base.copy()
            action[-1] = self.close_delta
            sequence.append(np.clip(action, -self.max_action_abs, self.max_action_abs))
        return sequence

    def run(self, env: Any, start_obs: Mapping[str, Any], action_v_t0: Sequence[float] | None = None) -> ProbeTrace:
        observations: List[Dict[str, Any]] = [dict(start_obs)]
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        infos: List[Dict[str, Any]] = []
        contact_counts: List[int] = [self._ncon(env)]
        done = False
        for action in self.make_sequence(action_v_t0):
            result = env.step(action)
            obs, reward, step_done, info = result.obs, result.reward, result.done, result.info
            observations.append(dict(obs))
            actions.append(action)
            rewards.append(float(reward))
            infos.append(dict(info))
            contact_counts.append(self._ncon(env))
            done = bool(step_done)
            if done:
                break
        return ProbeTrace(observations, actions, rewards, infos, contact_counts, done=done)

    @staticmethod
    def _ncon(env: Any) -> int:
        sim = getattr(env, "sim", getattr(getattr(env, "env", None), "sim", None))
        return int(getattr(getattr(sim, "data", None), "ncon", 0) or 0)


def trace_state_vectors(trace: ProbeTrace, cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "eef_pos": np.stack([extract_eef_pos(obs, cfg) for obs in trace.observations], axis=0),
        "object_pos": np.stack([extract_object_pos(obs, cfg) for obs in trace.observations], axis=0),
        "gripper": np.stack([extract_gripper(obs, cfg) for obs in trace.observations], axis=0),
        "contact_counts": np.asarray(trace.contact_counts, dtype=np.float32),
    }
