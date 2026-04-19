from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from env.libero_wrapper import extract_eef_pos, extract_object_pos
from rollout.rollout_utils import normalize_action


class MockPolicy:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.action_dim = int(cfg.get("policy", {}).get("action_dim", 7))

    def predict(self, obs: Mapping[str, Any], task_name: str) -> np.ndarray:
        eef = extract_eef_pos(obs, self.cfg)
        obj = extract_object_pos(obs, self.cfg)
        action = np.zeros(self.action_dim, dtype=np.float32)
        gripper = np.asarray(obs.get("robot0_gripper_qpos", obs.get("gripper", [0.08])), dtype=np.float32).reshape(-1)
        gripper_closed = bool(gripper.size and gripper.mean() < 0.04)
        basket = np.asarray(obs.get("basket_pos", [0.18, 0.20, 0.03]), dtype=np.float32)
        holding_like = gripper_closed and np.linalg.norm(eef[:2] - obj[:2]) < 0.07
        if holding_like or (gripper_closed and obj[2] > 0.06):
            target = np.array([eef[0], eef[1], 0.16], dtype=np.float32)
            if eef[2] > 0.13:
                target = np.array([basket[0], basket[1], 0.10], dtype=np.float32)
            if np.linalg.norm(eef[:2] - basket[:2]) < 0.06 and eef[2] > 0.08:
                target = np.array([basket[0], basket[1], 0.055], dtype=np.float32)
            action[:3] = np.clip(target - eef, -0.025, 0.025)
            action[-1] = -1.0
            return action
        delta = obj - eef
        if np.linalg.norm(delta) > 0.055:
            action[:3] = np.clip(delta, -0.025, 0.025)
            action[-1] = 1.0
        else:
            action[:3] = np.array([0.0, 0.0, -0.005], dtype=np.float32)
            action[-1] = -1.0
        return action


class OpenVLAPolicy:
    """Lazy OpenVLA adapter.

    This class intentionally keeps imports inside __init__ so the rest of the
    prototype can be tested without a GPU, MuJoCo, or Transformers installed.
    Different OpenVLA-OFT snapshots expose slightly different policy helpers;
    this adapter first tries a user-provided factory and then falls back to the
    Hugging Face AutoModel path.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.action_dim = int(cfg.get("policy", {}).get("action_dim", 7))
        self.checkpoint = str(cfg.get("checkpoint", "moojink/openvla-7b-oft-finetuned-libero-object"))
        self.device = str(cfg.get("policy", {}).get("device", "cuda"))
        self.policy = None
        self.processor = None
        factory_path = cfg.get("policy", {}).get("policy_factory")
        if factory_path:
            self.policy = self._load_factory(factory_path)(cfg)
            return
        self._load_hf_model()

    def _load_factory(self, path: str):
        import importlib

        module_name, function_name = path.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    def _load_hf_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "OpenVLA policy mode requires torch and transformers installed in "
                "/network/rit/lab/wang_lab_cs/yhan/envs. Use --mock for a local smoke test."
            ) from exc

        paths = self.cfg.get("paths", {})
        if paths.get("hf_home"):
            os.environ.setdefault("HF_HOME", str(paths["hf_home"]))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(paths["hf_home"]) / "transformers"))
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.policy = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            self.policy = self.policy.to(self.device)
        self.policy.eval()

    def predict(self, obs: Mapping[str, Any], task_name: str) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            return normalize_action(self.policy.predict(obs, task_name), self.action_dim)
        if hasattr(self.policy, "predict_action"):
            return normalize_action(self.policy.predict_action(obs, task_name), self.action_dim)
        return self._predict_hf(obs, task_name)

    def _predict_hf(self, obs: Mapping[str, Any], task_name: str) -> np.ndarray:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Hugging Face OpenVLA inference requires torch and pillow.") from exc
        from env.libero_wrapper import extract_image

        image = extract_image(obs, self.cfg)
        if image is None:
            raise RuntimeError("Observation does not contain an image for OpenVLA inference.")
        prompt = f"In: What action should the robot take to {task_name.replace('_', ' ')}?\nOut:"
        pil_image = Image.fromarray(np.asarray(image).astype(np.uint8))
        inputs = self.processor(prompt, pil_image).to(self.device)
        with torch.no_grad():
            if hasattr(self.policy, "predict_action"):
                action = self.policy.predict_action(**inputs)
            else:
                generated = self.policy.generate(**inputs, max_new_tokens=32)
                action = self.processor.decode(generated[0], skip_special_tokens=True)
        return normalize_action(action, self.action_dim)


def create_policy(cfg: Mapping[str, Any]):
    if cfg.get("policy", {}).get("mode", "mock") == "mock":
        return MockPolicy(cfg)
    return OpenVLAPolicy(cfg)


def fuse_actions(action_v: Sequence[float], action_r: Sequence[float] | None, alpha: float, action_dim: int) -> np.ndarray:
    action_v = normalize_action(action_v, action_dim)
    if action_r is None:
        return action_v
    action_r = normalize_action(action_r, action_dim)
    return (float(alpha) * action_r + (1.0 - float(alpha)) * action_v).astype(np.float32)
