from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from env.libero_wrapper import check_success, create_env, extract_image, get_libero_object_tasks
from probe.contact_detector import ContactDetector
from probe.feature_extractor import extract_probe_features
from probe.probe_runner import ProbeRunner
from retrieval.image_embedder import create_image_embedder
from retrieval.memory_bank import MemoryBank, MemoryItem
from rollout.rollout_utils import (
    add_common_args,
    append_jsonl,
    config_from_args,
    ensure_dir,
    seed_everything,
    write_json,
)
from vla.openvla_policy import create_policy


def collect_episode(
    task_name: str,
    trial_idx: int,
    cfg: Mapping[str, Any],
    output_dir: Path,
    memory: MemoryBank,
) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0)) + trial_idx
    env = create_env(task_name, cfg, seed=seed)
    policy = create_policy(cfg)
    detector = ContactDetector(cfg)
    probe = ProbeRunner(cfg)
    embedder = create_image_embedder(cfg)
    obs = env.reset()
    detector.reset()
    done = False
    success = False
    contact_triggered = False
    probe_start_step = None
    probe_end_step = None
    probe_features: Dict[str, float] = {}
    retrieved_candidate_ids: list[str] = []
    action_v_t0 = np.zeros(int(cfg.get("policy", {}).get("action_dim", 7)), dtype=np.float32)
    image_path = None
    post_probe_actions = []

    for step in range(int(cfg.get("max_steps", 220))):
        action_v = policy.predict(obs, task_name)
        event = detector.check(env, step)
        if event.triggered and not contact_triggered:
            contact_triggered = True
            probe_start_step = step
            action_v_t0 = action_v.astype(np.float32)
            image = extract_image(obs, cfg)
            if bool(cfg.get("logging", {}).get("save_debug_frames", True)) and image is not None:
                frame_dir = ensure_dir(output_dir / "frames")
                image_path = str(frame_dir / f"{task_name}_trial{trial_idx:03d}_contact.npy")
                np.save(image_path, image)
            trace = probe.run(env, obs, action_v_t0)
            probe_features = extract_probe_features(trace, cfg)
            probe_end_step = step + len(trace.actions)
            obs = trace.observations[-1]
            done = trace.done
            if done:
                success = any(check_success(info, reward) for info, reward in zip(trace.infos, trace.rewards))
                break
            for _ in range(3):
                post_probe_actions.append(policy.predict(obs, task_name).astype(np.float32))
            embedding = embedder.embed(image)
            memory.add(
                MemoryItem(
                    episode_id=f"{task_name}/trial_{trial_idx:04d}",
                    task_name=task_name,
                    image_embedding=embedding.astype(np.float32),
                    raw_image_path=image_path,
                    action_v_t0=action_v_t0,
                    probe_features=probe_features,
                    post_probe_action_chunk=np.asarray(post_probe_actions, dtype=np.float32) if post_probe_actions else None,
                    success=False,
                    metadata={
                        "seed": seed,
                        "probe_start_step": probe_start_step,
                        "probe_end_step": probe_end_step,
                        "contact_event": event.__dict__,
                    },
                )
            )
        result = env.step(action_v)
        obs = result.obs
        done = result.done
        success = check_success(result.info, result.reward)
        if done:
            break

    if memory.items and memory.items[-1].episode_id == f"{task_name}/trial_{trial_idx:04d}":
        memory.items[-1].success = bool(success)
        memory.items[-1].metadata["final_success"] = bool(success)

    env.close()
    return {
        "episode_id": f"{task_name}/trial_{trial_idx:04d}",
        "task_name": task_name,
        "seed": seed,
        "contact_triggered": contact_triggered,
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "probe_features": probe_features,
        "retrieved_candidate_ids": retrieved_candidate_ids,
        "fused_action_stats": None,
        "success": bool(success),
    }


def run_collection(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if cfg.get("env", {}).get("mode") == "real" or cfg.get("policy", {}).get("mode") == "openvla":
        from rollout.real_libero import run_real_collection

        return run_real_collection(cfg)
    seed_everything(int(cfg.get("seed", 0)))
    output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "artifacts"))
    memory_dir = ensure_dir(cfg.get("paths", {}).get("memory_dir", output_dir / "memory_bank"))
    log_path = output_dir / "collection_episodes.jsonl"
    memory = MemoryBank()
    task_results = []
    for task_name in get_libero_object_tasks():
        for trial_idx in range(int(cfg.get("num_trials_per_task", 1))):
            row = collect_episode(task_name, trial_idx, cfg, output_dir, memory)
            task_results.append(row)
            append_jsonl(log_path, row)
    memory.save(memory_dir)
    summary = {
        "num_items": len(memory),
        "num_episodes": len(task_results),
        "success_rate": float(np.mean([r["success"] for r in task_results])) if task_results else 0.0,
        "memory_dir": str(memory_dir),
        "log_path": str(log_path),
    }
    write_json(output_dir / "collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect contact-probe memory for LIBERO object tasks.")
    add_common_args(parser, default_config="collect_libero_object.yaml")
    args, unknown = parser.parse_known_args()
    cfg = config_from_args(args, unknown)
    summary = run_collection(cfg)
    print(summary)


if __name__ == "__main__":
    main()
