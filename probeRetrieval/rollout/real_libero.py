from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from env.libero_wrapper import check_success
from probe.contact_detector import ContactDetector
from probe.feature_extractor import extract_probe_features
from probe.probe_runner import ProbeTrace
from retrieval.image_embedder import create_image_embedder
from retrieval.image_retrieval import retrieve_top_k
from retrieval.memory_bank import MemoryBank, MemoryItem
from retrieval.probe_rerank import aggregate_retrieved_action, rerank_by_probe
from rollout.rollout_utils import append_jsonl, ensure_dir, seed_everything, write_json
from rollout.video_utils import FrameBuffer, extend_frames, should_save_video, write_video
from vla.openvla_policy import fuse_actions


def setup_real_paths(cfg: Mapping[str, Any]) -> None:
    paths = cfg.get("paths", {})
    openvla_repo = str(paths.get("openvla_oft_repo") or "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft")
    libero_repo = str(paths.get("libero_repo") or "/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO")
    hf_home = "/network/rit/lab/wang_lab_cs/yhan/hf_cache"
    cache_home = "/network/rit/lab/wang_lab_cs/yhan/cache"
    libero_config = "/network/rit/lab/wang_lab_cs/yhan/.libero"
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(cache_home, exist_ok=True)
    os.makedirs(os.path.join(cache_home, "matplotlib"), exist_ok=True)
    os.makedirs(libero_config, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
    os.environ["XDG_CACHE_HOME"] = cache_home
    os.environ["MPLCONFIGDIR"] = os.path.join(cache_home, "matplotlib")
    os.environ["LIBERO_CONFIG_PATH"] = libero_config
    config_path = os.path.join(libero_config, "config.yaml")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        f"assets: {libero_repo}/libero/libero/assets",
                        f"bddl_files: {libero_repo}/libero/libero/bddl_files",
                        f"benchmark_root: {libero_repo}/libero/libero",
                        f"datasets: {libero_repo}/libero/datasets",
                        f"init_states: {libero_repo}/libero/libero/init_files",
                        "",
                    ]
                )
            )
    for path in (libero_repo, openvla_repo):
        if path not in sys.path:
            sys.path.insert(0, path)


def make_generate_config(cfg: Mapping[str, Any]):
    setup_real_paths(cfg)
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    return GenerateConfig(
        pretrained_checkpoint=str(cfg.get("checkpoint", "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-7b-oft")),
        task_suite_name=str(cfg.get("task_suite_name", "libero_object")),
        center_crop=bool(cfg.get("center_crop", True)),
        num_trials_per_task=int(cfg.get("num_trials_per_task", 1)),
        seed=int(cfg.get("seed", 0)),
        local_log_dir=str(Path(cfg.get("paths", {}).get("output_dir", "artifacts")) / "openvla_logs"),
        num_open_loop_steps=int(cfg.get("num_open_loop_steps", NUM_ACTIONS_CHUNK)),
        num_images_in_input=int(cfg.get("num_images_in_input", 1)),
        use_proprio=bool(cfg.get("use_proprio", False)),
        use_l1_regression=bool(cfg.get("use_l1_regression", False)),
        use_diffusion=bool(cfg.get("use_diffusion", False)),
        load_in_8bit=bool(cfg.get("policy", {}).get("load_in_8bit", False)),
        load_in_4bit=bool(cfg.get("policy", {}).get("load_in_4bit", False)),
    )


def initialize_real_policy(cfg: Mapping[str, Any]):
    setup_real_paths(cfg)
    from experiments.robot.libero.run_libero_eval import initialize_model
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    real_cfg = make_generate_config(cfg)
    set_seed_everywhere(real_cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(real_cfg)
    resize_size = get_image_resize_size(real_cfg)
    return real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector


def get_real_action_chunk(real_cfg, model, obs, task_description, resize_size, processor, action_head, proprio_projector, noisy_action_projector):
    from experiments.robot.libero.run_libero_eval import prepare_observation, process_action
    from experiments.robot.robot_utils import get_action

    observation, img = prepare_observation(obs, resize_size)
    raw_actions = get_action(
        real_cfg,
        model,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=real_cfg.use_film,
    )
    actions = [process_action(np.asarray(action, dtype=np.float32), real_cfg.model_family).astype(np.float32) for action in raw_actions]
    return actions, img


def sim_ncon(env: Any) -> int:
    return int(getattr(getattr(getattr(env, "sim", None), "data", None), "ncon", 0) or 0)


def real_probe_actions(cfg: Mapping[str, Any]) -> list[np.ndarray]:
    action_dim = int(cfg.get("policy", {}).get("action_dim", 7))
    probe_cfg = cfg.get("probe", {})
    close_steps = int(probe_cfg.get("num_close_steps", 2))
    lift_steps = int(probe_cfg.get("num_lift_steps", 3))
    hold_steps = int(probe_cfg.get("num_hold_steps", 1))
    lift_delta_z = float(probe_cfg.get("lift_delta_z", 0.015))
    sequence: list[np.ndarray] = []
    for _ in range(close_steps):
        action = np.zeros(action_dim, dtype=np.float32)
        action[-1] = 1.0
        sequence.append(action)
    for _ in range(lift_steps):
        action = np.zeros(action_dim, dtype=np.float32)
        action[2] = lift_delta_z
        action[-1] = 1.0
        sequence.append(action)
    for _ in range(hold_steps):
        action = np.zeros(action_dim, dtype=np.float32)
        action[-1] = 1.0
        sequence.append(action)
    return sequence


def run_real_probe(env: Any, start_obs: Mapping[str, Any], cfg: Mapping[str, Any]) -> ProbeTrace:
    observations = [dict(start_obs)]
    actions = []
    rewards = []
    infos = []
    contact_counts = [sim_ncon(env)]
    done = False
    for action in real_probe_actions(cfg):
        obs, reward, step_done, info = env.step(action.tolist())
        observations.append(dict(obs))
        actions.append(action)
        rewards.append(float(reward))
        infos.append(dict(info or {}))
        contact_counts.append(sim_ncon(env))
        done = bool(step_done)
        if done:
            break
    return ProbeTrace(observations, actions, rewards, infos, contact_counts, done=done)


def first_object_position(obs: Mapping[str, Any]) -> np.ndarray:
    for key, value in obs.items():
        if key.endswith("_pos") and not key.startswith("robot0") and "eef" not in key and "camera" not in key:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
    return np.zeros(3, dtype=np.float32)


def augment_obs_for_features(obs: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(obs)
    if "target_object_pos" not in out:
        out["target_object_pos"] = first_object_position(obs)
    return out


def trace_with_augmented_obs(trace: ProbeTrace) -> ProbeTrace:
    return ProbeTrace(
        observations=[augment_obs_for_features(obs) for obs in trace.observations],
        actions=trace.actions,
        rewards=trace.rewards,
        infos=trace.infos,
        contact_counts=trace.contact_counts,
        done=trace.done,
    )


def load_task_suite(real_cfg):
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[real_cfg.task_suite_name]()
    return task_suite


def make_env_for_task(task, real_cfg):
    from experiments.robot.libero.libero_utils import get_libero_env

    return get_libero_env(task, real_cfg.model_family, resolution=real_cfg.env_img_res)


def get_initial_state(real_cfg, task_suite, task_id: int, episode_idx: int):
    initial_states = task_suite.get_task_init_states(task_id)
    return initial_states[episode_idx]


def run_real_collection(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    seed_everything(int(cfg.get("seed", 0)))
    output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "artifacts/real_collect"))
    memory_dir = ensure_dir(cfg.get("paths", {}).get("memory_dir", output_dir / "memory_bank"))
    log_path = output_dir / "collection_episodes.jsonl"
    real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector = initialize_real_policy(cfg)
    task_suite = load_task_suite(real_cfg)
    embedder = create_image_embedder(cfg)
    memory = MemoryBank()
    rows = []
    max_steps = int(cfg.get("max_steps", 280))
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env, task_description = make_env_for_task(task, real_cfg)
        for episode_idx in range(real_cfg.num_trials_per_task):
            row = collect_real_episode(
                cfg,
                real_cfg,
                env,
                task_suite,
                task_id,
                episode_idx,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                embedder,
                memory,
                output_dir,
                max_steps,
            )
            rows.append(row)
            append_jsonl(log_path, row)
        env.close()
    memory.save(memory_dir)
    summary = {
        "num_items": len(memory),
        "num_episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "memory_dir": str(memory_dir),
        "log_path": str(log_path),
    }
    write_json(output_dir / "collection_summary.json", summary)
    return summary


def collect_real_episode(
    cfg,
    real_cfg,
    env,
    task_suite,
    task_id,
    episode_idx,
    task_description,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    embedder,
    memory,
    output_dir,
    max_steps,
):
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation

    env.reset()
    obs = env.set_init_state(get_initial_state(real_cfg, task_suite, task_id, episode_idx))
    for _ in range(real_cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(real_cfg.model_family))
    action_queue: deque[np.ndarray] = deque(maxlen=real_cfg.num_open_loop_steps)
    contact_triggered = False
    probe_features: Dict[str, float] = {}
    probe_start_step = None
    probe_end_step = None
    action_v_t0 = np.zeros(7, dtype=np.float32)
    contact_image = None
    post_probe_actions = []
    success = False
    detector = ContactDetector(cfg)
    detector.reset()
    log_cfg = cfg.get("logging", {})
    save_videos = bool(log_cfg.get("save_videos", False))
    video_every = int(log_cfg.get("video_every", 0))
    video_fps = int(log_cfg.get("video_fps", 20))
    record_video = save_videos and should_save_video(episode_idx, max(video_every, 1))
    full_frames: list[np.ndarray] = []
    pre_probe_frames = FrameBuffer(maxlen=int(log_cfg.get("probe_clip_pre_frames", video_fps)))
    probe_frames: list[np.ndarray] = []
    save_probe_clips = bool(log_cfg.get("save_probe_clips", True))
    for step in range(max_steps):
        observation, img = prepare_observation(obs, resize_size)
        if record_video:
            extend_frames(full_frames, [img])
            pre_probe_frames.append(img)
        if len(action_queue) == 0:
            action_queue.extend(
                get_real_action_chunk(
                    real_cfg, model, obs, task_description, resize_size, processor, action_head, proprio_projector, noisy_action_projector
                )[0]
            )
        action = np.asarray(action_queue.popleft(), dtype=np.float32)
        event = detector.check(env, step)
        if not contact_triggered and event.triggered:
            contact_triggered = True
            probe_start_step = step
            probe_end_step = step + len(real_probe_actions(cfg))
            action_v_t0 = action.copy()
            contact_image = img.copy()
            trace = trace_with_augmented_obs(run_real_probe(env, augment_obs_for_features(obs), cfg))
            probe_features = extract_probe_features(trace, cfg)
            if record_video:
                extend_frames(probe_frames, [obs_i.get("agentview_image")[::-1, ::-1] for obs_i in trace.observations if "agentview_image" in obs_i])
            obs = trace.observations[-1]
            action_queue.clear()
            post_probe_actions = get_real_action_chunk(
                real_cfg, model, obs, task_description, resize_size, processor, action_head, proprio_projector, noisy_action_projector
            )[0][:3]
            if trace.done:
                success = True
                break
            continue
        obs, reward, done, info = env.step(action.tolist())
        if done or check_success(info, reward):
            success = True
            break
    if contact_triggered:
        image_path = None
        video_path = None
        probe_video_path = None
        if contact_image is not None and bool(cfg.get("logging", {}).get("save_debug_frames", True)):
            frame_dir = ensure_dir(output_dir / "frames")
            safe_task = task_description.replace(" ", "_")[:80]
            image_path = str(frame_dir / f"{task_id:02d}_{episode_idx:04d}_{safe_task}.npy")
            np.save(image_path, contact_image)
        if record_video and full_frames:
            video_dir = ensure_dir(output_dir / "videos")
            safe_task = task_description.replace(" ", "_")[:80]
            video_path = write_video(video_dir / f"{task_id:02d}_{episode_idx:04d}_{safe_task}.mp4", full_frames, fps=video_fps)
            if save_probe_clips and probe_frames:
                probe_clip = pre_probe_frames.to_list() + probe_frames
                probe_video_path = write_video(
                    video_dir / f"{task_id:02d}_{episode_idx:04d}_{safe_task}_probe.mp4",
                    probe_clip,
                    fps=video_fps,
                )
        memory.add(
            MemoryItem(
                episode_id=f"task_{task_id:02d}/trial_{episode_idx:04d}",
                task_name=task_description,
                image_embedding=embedder.embed(contact_image).astype(np.float32),
                raw_image_path=image_path,
                action_v_t0=action_v_t0,
                probe_features=probe_features,
                post_probe_action_chunk=np.asarray(post_probe_actions, dtype=np.float32) if post_probe_actions else None,
                success=bool(success),
                metadata={
                    "task_id": task_id,
                    "probe_start_step": probe_start_step,
                    "probe_end_step": probe_end_step,
                    "contact_event": event.__dict__,
                    "video_path": str(video_path) if video_path else None,
                    "probe_video_path": str(probe_video_path) if probe_video_path else None,
                },
            )
        )
    return {
        "episode_id": f"task_{task_id:02d}/trial_{episode_idx:04d}",
        "task_name": task_description,
        "task_id": task_id,
        "seed": real_cfg.seed,
        "contact_triggered": contact_triggered,
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "probe_features": probe_features,
        "retrieved_candidate_ids": [],
        "fused_action_stats": None,
        "success": bool(success),
    }


def run_real_inference(cfg: Mapping[str, Any], variants: Sequence[str]) -> Dict[str, Any]:
    output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "artifacts/real_eval"))
    real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector = initialize_real_policy(cfg)
    task_suite = load_task_suite(real_cfg)
    embedder = create_image_embedder(cfg)
    memory_dir = cfg.get("paths", {}).get("memory_dir", output_dir / "memory_bank")
    bank = MemoryBank.load(memory_dir) if any(v != "baseline_vla" for v in variants) else None
    rows = []
    max_steps = int(cfg.get("max_steps", 280))
    for variant in variants:
        log_path = output_dir / f"{variant}_episodes.jsonl"
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = make_env_for_task(task, real_cfg)
            for episode_idx in range(real_cfg.num_trials_per_task):
                row = infer_real_episode(
                    cfg,
                    real_cfg,
                    env,
                    task_suite,
                    task_id,
                    episode_idx,
                    task_description,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    embedder,
                    bank,
                    variant,
                    max_steps,
                )
                rows.append(row)
                append_jsonl(log_path, row)
            env.close()
    summary = summarize_variant_rows(rows)
    write_json(output_dir / "inference_summary.json", summary)
    return summary


def infer_real_episode(
    cfg,
    real_cfg,
    env,
    task_suite,
    task_id,
    episode_idx,
    task_description,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    embedder,
    bank,
    variant,
    max_steps,
):
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation

    env.reset()
    obs = env.set_init_state(get_initial_state(real_cfg, task_suite, task_id, episode_idx))
    for _ in range(real_cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(real_cfg.model_family))
    action_queue: deque[np.ndarray] = deque(maxlen=real_cfg.num_open_loop_steps)
    contact_triggered = False
    probe_features: Dict[str, float] = {}
    probe_start_step = None
    probe_end_step = None
    action_r = None
    ranked = []
    retrieved_ids: list[str] = []
    fused_norms: list[float] = []
    fusion_start_step = None
    success = False
    detector = ContactDetector(cfg)
    detector.reset()
    for step in range(max_steps):
        observation, img = prepare_observation(obs, resize_size)
        if len(action_queue) == 0:
            action_queue.extend(
                get_real_action_chunk(
                    real_cfg, model, obs, task_description, resize_size, processor, action_head, proprio_projector, noisy_action_projector
                )[0]
            )
        action_v = np.asarray(action_queue.popleft(), dtype=np.float32)
        action = action_v
        event = detector.check(env, step)
        if variant != "baseline_vla" and not contact_triggered and event.triggered:
            contact_triggered = True
            probe_start_step = step
            top = retrieve_top_k(
                embedder.embed(img),
                bank,
                int(cfg.get("retrieval", {}).get("top_k", 10)),
                eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
            )
            if variant == "full_probe_rerank":
                trace = trace_with_augmented_obs(run_real_probe(env, augment_obs_for_features(obs), cfg))
                probe_features = extract_probe_features(trace, cfg)
                probe_end_step = step + len(trace.actions)
                obs = trace.observations[-1]
                action_queue.clear()
                ranked = rerank_by_probe(
                    probe_features,
                    top,
                    bank,
                    int(cfg.get("retrieval", {}).get("rerank_top_k", 5)),
                    image_weight=float(cfg.get("retrieval", {}).get("image_weight", 0.35)),
                    probe_weight=float(cfg.get("retrieval", {}).get("probe_weight", 0.65)),
                    eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                )
                if trace.done:
                    success = True
                    break
                continue
            else:
                ranked = top
            action_r, retrieved_ids = aggregate_retrieved_action(
                ranked,
                bank,
                successful_only=bool(cfg.get("retrieval", {}).get("successful_only", True)),
                eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
            )
            fusion_start_step = step
        if action_r is not None and fusion_start_step is not None:
            if step - fusion_start_step < int(cfg.get("fusion", {}).get("max_steps", 30)):
                action = fuse_actions(action_v, action_r, float(cfg.get("fusion", {}).get("alpha", 0.5)), 7)
                fused_norms.append(float(np.linalg.norm(action - action_v)))
        obs, reward, done, info = env.step(action.tolist())
        if done or check_success(info, reward):
            success = True
            break
    return {
        "episode_id": f"{variant}/task_{task_id:02d}/trial_{episode_idx:04d}",
        "variant": variant,
        "task_name": task_description,
        "task_id": task_id,
        "seed": real_cfg.seed,
        "contact_triggered": contact_triggered,
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "probe_features": probe_features,
        "retrieved_candidate_ids": retrieved_ids,
        "retrieved_candidates": ranked,
        "fused_action_stats": {
            "count": len(fused_norms),
            "mean_delta_norm": float(np.mean(fused_norms)) if fused_norms else 0.0,
            "max_delta_norm": float(np.max(fused_norms)) if fused_norms else 0.0,
        },
        "success": bool(success),
    }


def summarize_variant_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"variants": {}}
    for variant in sorted({str(r["variant"]) for r in rows}):
        vrows = [r for r in rows if r["variant"] == variant]
        per_task = {}
        for task in sorted({str(r["task_name"]) for r in vrows}):
            trows = [r for r in vrows if r["task_name"] == task]
            per_task[task] = {"success_rate": float(np.mean([r["success"] for r in trows])), "n": len(trows)}
        summary["variants"][variant] = {
            "average_success_rate": float(np.mean([r["success"] for r in vrows])) if vrows else 0.0,
            "n": len(vrows),
            "per_task": per_task,
        }
    return summary
