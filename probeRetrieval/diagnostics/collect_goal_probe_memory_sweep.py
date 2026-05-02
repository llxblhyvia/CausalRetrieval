from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from probe.contact_detector import ContactDetector
from retrieval.image_embedder import create_image_embedder
from retrieval.memory_bank import MemoryBank, MemoryItem
from rollout.real_libero import (
    get_initial_state,
    get_real_action_chunk,
    initialize_real_policy,
    load_task_suite,
    make_env_for_task,
    setup_real_paths,
)
from rollout.rollout_utils import append_jsonl, ensure_dir, write_json

from diagnostics.probe_goal_physics_sweep import (
    DEFAULT_SELECTED_TASK_SPECS,
    apply_physics_modifiers,
    body_position,
    body_quaternion,
    build_setting_rows,
    capture_physics_snapshot,
    contact_active,
    extract_agentview,
    get_end_effector_position,
    make_detector_cfg,
    mean_std_max,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    quat_geodesic_degrees,
    resolve_task_handles,
    resolve_task_ids,
    restore_physics_snapshot,
    run_probe,
    sanitize_name,
    summarize_rows,
    target_position,
    translational_force_norm,
)


def run_single_episode(
    *,
    env: Any,
    task_suite: Any,
    task_id: int,
    task_description: str,
    episode_idx: int,
    setting: Mapping[str, Any],
    real_cfg: Any,
    model: Any,
    resize_size: int,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    noisy_action_projector: Any,
    embedder: Any,
    memory: MemoryBank,
    output_dir: Path,
    cfg: Mapping[str, Any],
    max_steps: int,
) -> Dict[str, Any]:
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation

    env.reset()
    obs = env.set_init_state(get_initial_state(real_cfg, task_suite, task_id, episode_idx))
    for _ in range(real_cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(real_cfg.model_family))

    handles = resolve_task_handles(env, task_suite.get_task(task_id))
    detector = ContactDetector(make_detector_cfg(cfg, handles.get("target_instance")))
    detector.reset()
    snapshot = capture_physics_snapshot(env)
    applied = apply_physics_modifiers(
        env,
        handles,
        friction_value=setting.get("friction_value"),
        mass_scale=setting.get("mass_scale"),
    )

    action_queue: deque[np.ndarray] = deque(maxlen=real_cfg.num_open_loop_steps)
    ee_positions = [get_end_effector_position(obs)]
    object_positions = [body_position(env, handles.get("target_body_id"))]
    object_quats = [body_quaternion(env, handles.get("target_body_id"))]
    target_pos = target_position(env, handles)
    force_values: list[float] = []
    contact_flags: list[bool] = []
    probe_triggered = False
    probe_start_step = None
    probe_end_step = None
    probe_metrics = {
        "probe_num_steps": 0,
        "probe_obj_displacement": 0.0,
        "probe_total_obj_motion": 0.0,
        "probe_max_object_step_displacement": 0.0,
        "probe_object_rotation_change_deg": 0.0,
        "probe_ee_motion": 0.0,
        "probe_contact_steps": 0,
        "probe_contact_ratio": 0.0,
        "post_probe_object_to_target_distance": None,
        "probe_mean_force": 0.0,
        "probe_max_force": 0.0,
        "probe_force_std": 0.0,
        "probe_force_values": [],
        "probe_contact_flags": [],
    }
    success = False
    contact_image = None
    image_path = None
    action_v_t0 = np.zeros(int(cfg.get("policy", {}).get("action_dim", 7)), dtype=np.float32)
    post_probe_actions: list[np.ndarray] = []
    probe_features: Dict[str, float] = {}

    try:
        for step in range(int(max_steps)):
            observation, img = prepare_observation(obs, resize_size)
            frame = extract_agentview(obs)
            is_contact = contact_active(detector, env)
            contact_flags.append(bool(is_contact))
            force_values.append(translational_force_norm(env, handles.get("target_body_id")))
            if len(action_queue) == 0:
                action_queue.extend(
                    get_real_action_chunk(
                        real_cfg,
                        model,
                        obs,
                        task_description,
                        resize_size,
                        processor,
                        action_head,
                        proprio_projector,
                        noisy_action_projector,
                    )[0]
                )
            action_v = np.asarray(action_queue.popleft(), dtype=np.float32)
            event = detector.check(env, step)
            if event.triggered and not probe_triggered:
                probe_triggered = True
                probe_start_step = step
                probe_end_step = step + len(cfg.get("probe", {}))
                action_v_t0 = action_v.copy()
                contact_image = img.copy()
                if bool(cfg.get("logging", {}).get("save_debug_frames", True)):
                    frame_dir = ensure_dir(output_dir / "frames")
                    safe_task = sanitize_name(task_description)[:80]
                    image_path = str(frame_dir / f"task{task_id:02d}_ep{episode_idx:03d}_{sanitize_name(setting['setting_name'])}_{safe_task}.npy")
                    np.save(image_path, contact_image)
                obs, probe_result, _probe_frames = run_probe(env, cfg, handles, detector, obs)
                probe_features = {
                    "probe_start_step": float(step),
                    "contact_steps": float(probe_result["probe_contact_steps"]),
                    "contact_ratio": float(probe_result["probe_contact_ratio"]),
                    "mean_force": float(probe_result["probe_mean_force"]),
                    "max_force": float(probe_result["probe_max_force"]),
                    "force_std": float(probe_result["probe_force_std"]),
                    "probe_obj_displacement": float(probe_result["probe_obj_displacement"]),
                    "end_effector_movement": float(probe_result["probe_ee_motion"]),
                    "post_probe_object_to_target_distance": float(probe_result["post_probe_object_to_target_distance"]),
                }
                probe_end_step = step + int(probe_result["probe_num_steps"])
                probe_metrics.update(probe_result)
                ee_positions.append(get_end_effector_position(obs))
                object_positions.append(body_position(env, handles.get("target_body_id")))
                object_quats.append(body_quaternion(env, handles.get("target_body_id")))
                contact_flags.extend([bool(flag) for flag in probe_result["probe_contact_flags"]])
                force_values.extend([float(value) for value in probe_result["probe_force_values"]])
                action_queue.clear()
                if not probe_result["done"]:
                    post_probe_actions = get_real_action_chunk(
                        real_cfg,
                        model,
                        obs,
                        task_description,
                        resize_size,
                        processor,
                        action_head,
                        proprio_projector,
                        noisy_action_projector,
                    )[0][:3]
                if probe_result["done"]:
                    success = bool(probe_result["success"])
                    break
                continue

            obs, reward, done, info = env.step(action_v.tolist())
            ee_positions.append(get_end_effector_position(obs))
            object_positions.append(body_position(env, handles.get("target_body_id")))
            object_quats.append(body_quaternion(env, handles.get("target_body_id")))
            if done or bool(info.get("success")) or reward > 0.5:
                success = True
                break
    finally:
        restore_physics_snapshot(env, snapshot)

    ee_positions_arr = np.stack(ee_positions, axis=0)
    object_positions_arr = np.stack(object_positions, axis=0)
    object_step_displacements = (
        np.linalg.norm(np.diff(object_positions_arr, axis=0), axis=1) if len(object_positions_arr) > 1 else np.zeros(0)
    )
    ee_step_displacements = (
        np.linalg.norm(np.diff(ee_positions_arr, axis=0), axis=1) if len(ee_positions_arr) > 1 else np.zeros(0)
    )
    mean_force, force_std, max_force = mean_std_max(force_values)
    total_ee_motion = float(ee_step_displacements.sum()) if ee_step_displacements.size else 0.0
    total_obj_motion = float(object_step_displacements.sum()) if object_step_displacements.size else 0.0
    final_target_distance = float(np.linalg.norm(object_positions_arr[-1] - target_pos))
    episode_id = f"task_{task_id:02d}/trial_{episode_idx:04d}"

    if probe_triggered and contact_image is not None:
        memory.add(
            MemoryItem(
                episode_id=episode_id,
                task_name=task_description,
                image_embedding=embedder.embed(contact_image).astype(np.float32),
                raw_image_path=image_path,
                action_v_t0=action_v_t0.astype(np.float32),
                probe_features={k: float(v) for k, v in probe_features.items()},
                post_probe_action_chunk=np.asarray(post_probe_actions, dtype=np.float32) if post_probe_actions else None,
                success=bool(success),
                metadata={
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "setting_name": str(setting["setting_name"]),
                    "sweep_type": str(setting["sweep_type"]),
                    "friction_value": applied["friction_value"],
                    "mass_scale": applied["mass_scale"],
                    "probe_start_step": probe_start_step,
                    "probe_end_step": probe_end_step,
                },
            )
        )

    return {
        "episode_id": episode_id,
        "task_id": int(task_id),
        "task_name": task_description,
        "episode_idx": int(episode_idx),
        "sweep_type": setting["sweep_type"],
        "setting_name": setting["setting_name"],
        "friction_value": applied["friction_value"],
        "mass_scale": applied["mass_scale"],
        "mean_force": mean_force,
        "max_force": max_force,
        "force_std": force_std,
        "contact_ratio": float(np.mean(contact_flags)) if contact_flags else 0.0,
        "total_ee_motion": total_ee_motion,
        "total_obj_motion": total_obj_motion,
        "motion_ratio": float(total_obj_motion / (total_ee_motion + 1.0e-6)),
        "probe_obj_displacement": float(probe_metrics["probe_obj_displacement"]),
        "object_rotation_change_deg": float(probe_metrics["probe_object_rotation_change_deg"]),
        "end_effector_movement": float(probe_metrics["probe_ee_motion"]),
        "contact_steps": int(probe_metrics["probe_contact_steps"]),
        "max_object_step_displacement": float(
            max(
                float(object_step_displacements.max()) if object_step_displacements.size else 0.0,
                float(probe_metrics["probe_max_object_step_displacement"]),
            )
        ),
        "final_success": bool(success),
        "post_probe_object_to_target_distance": float(probe_metrics["post_probe_object_to_target_distance"])
        if probe_metrics["post_probe_object_to_target_distance"] is not None
        else None,
        "probe_triggered": bool(probe_triggered),
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "final_object_to_target_distance": final_target_distance,
        "target_instance": handles.get("target_instance"),
        "target_reference": handles.get("target_reference"),
        "target_body_names": handles.get("target_body_names", []),
        "target_site_names": handles.get("target_site_names", []),
        "video_path": None,
        "probe_video_path": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect full memory bank for LIBERO-goal probe sweep.")
    parser.add_argument("--output_dir", default="artifacts/collect_goal_probe_memory_sweep")
    parser.add_argument("--memory_dir", default=None)
    parser.add_argument("--task_ids", default=None)
    parser.add_argument("--task_names", default=None)
    parser.add_argument("--frictions", default="0.2,0.7")
    parser.add_argument("--mass_scales", default="0.5,3.0,7.0")
    parser.add_argument("--episodes_per_setting", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {
        "checkpoint": "openvla/openvla-7b-finetuned-libero-goal",
        "task_suite_name": "libero_goal",
        "center_crop": True,
        "num_trials_per_task": int(args.episodes_per_setting),
        "num_images_in_input": 1,
        "use_proprio": False,
        "use_l1_regression": False,
        "use_diffusion": False,
        "num_open_loop_steps": 1,
        "seed": 0,
        "paths": {
            "output_dir": str(Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval") / args.output_dir),
            "memory_dir": str(
                Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval")
                / (args.memory_dir or f"{args.output_dir}/memory_bank")
            ),
            "openvla_oft_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft",
            "libero_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO",
        },
        "policy": {"load_in_8bit": False, "load_in_4bit": False, "action_dim": 7},
        "env": {"mode": "real"},
        "contact": {
            "use_filtered_contacts": True,
            "gripper_name_patterns": ["gripper", "finger", "leftpad", "rightpad"],
            "target_name_patterns": [],
            "min_contact_steps": 1,
        },
        "probe": {
            "num_close_steps": 2,
            "num_lift_steps": 3,
            "num_hold_steps": 1,
            "lift_delta_z": 0.015,
        },
        "logging": {"save_debug_frames": True},
    }

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    memory_dir = ensure_dir(cfg["paths"]["memory_dir"])
    jsonl_path = output_dir / "episodes.jsonl"

    setup_real_paths(cfg)
    real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector = initialize_real_policy(cfg)
    real_cfg.num_open_loop_steps = 1
    real_cfg.num_trials_per_task = int(args.episodes_per_setting)
    task_suite = load_task_suite(real_cfg)
    task_ids = resolve_task_ids(
        task_suite,
        requested_specs=parse_csv_strings(args.task_names) or DEFAULT_SELECTED_TASK_SPECS,
        requested_ids=parse_csv_ints(args.task_ids),
    )
    embedder = create_image_embedder(cfg)
    memory = MemoryBank()

    settings = build_setting_rows(parse_csv_floats(args.frictions), parse_csv_floats(args.mass_scales))
    for setting in settings:
        if setting["sweep_type"] == "friction":
            setting["setting_name"] = f"friction_mu_{setting['friction_value']:g}"
        else:
            setting["setting_name"] = f"mass_scale_{setting['mass_scale']:g}"

    rows = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env, task_description = make_env_for_task(task, real_cfg)
        try:
            for setting in settings:
                for episode_idx in range(int(args.episodes_per_setting)):
                    row = run_single_episode(
                        env=env,
                        task_suite=task_suite,
                        task_id=task_id,
                        task_description=task_description,
                        episode_idx=episode_idx,
                        setting=setting,
                        real_cfg=real_cfg,
                        model=model,
                        resize_size=resize_size,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        embedder=embedder,
                        memory=memory,
                        output_dir=output_dir,
                        cfg=cfg,
                        max_steps=int(args.max_steps),
                    )
                    rows.append(row)
                    append_jsonl(jsonl_path, row)
        finally:
            env.close()

    memory.save(memory_dir)
    write_json(output_dir / "summary.json", summarize_rows(rows))
    write_json(
        output_dir / "collection_summary.json",
        {
            "num_items": len(memory),
            "num_episodes": len(rows),
            "success_rate": float(np.mean([bool(r["final_success"]) for r in rows])) if rows else 0.0,
            "memory_dir": str(memory_dir),
            "episodes_path": str(jsonl_path),
        },
    )


if __name__ == "__main__":
    main()
