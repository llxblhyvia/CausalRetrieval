from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from probe.contact_detector import ContactDetector
from rollout.real_libero import (
    get_initial_state,
    get_real_action_chunk,
    initialize_real_policy,
    load_task_suite,
    make_env_for_task,
    setup_real_paths,
)
from rollout.rollout_utils import append_jsonl, ensure_dir, write_json
from rollout.video_utils import should_save_video, write_video

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
    quat_geodesic_degrees,
    resolve_task_handles,
    resolve_task_ids,
    restore_physics_snapshot,
    sanitize_name,
    summarize_rows,
    target_position,
    translational_force_norm,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
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
    cfg: Mapping[str, Any],
    max_steps: int,
    save_video: bool,
    video_dir: Path,
    video_fps: int,
) -> Dict[str, Any]:
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

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
    frames: list[np.ndarray] = []
    ee_positions = [get_end_effector_position(obs)]
    object_positions = [body_position(env, handles.get("target_body_id"))]
    object_quats = [body_quaternion(env, handles.get("target_body_id"))]
    target_pos = target_position(env, handles)
    force_values: list[float] = []
    contact_flags: list[bool] = []
    success = False

    try:
        for _step in range(int(max_steps)):
            frame = extract_agentview(obs)
            if save_video and frame is not None:
                frames.append(frame)

            contact_flags.append(bool(contact_active(detector, env)))
            force_values.append(translational_force_norm(env, handles.get("target_body_id")))

            if len(action_queue) == 0:
                actions, _ = get_real_action_chunk(
                    real_cfg,
                    model,
                    obs,
                    task_description,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                )
                action_queue.extend(actions)

            action = np.asarray(action_queue.popleft(), dtype=np.float32)
            obs, reward, done, info = env.step(action.tolist())
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
    total_rotation_change_deg = quat_geodesic_degrees(object_quats[0], object_quats[-1])

    video_path = None
    if save_video and frames:
        safe_task = sanitize_name(task_description)[:80]
        stem = f"task{task_id:02d}_ep{episode_idx:03d}_{sanitize_name(setting['setting_name'])}_{safe_task}"
        video_path = write_video(video_dir / f"{stem}.mp4", frames, fps=video_fps)

    return {
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
        "probe_obj_displacement": 0.0,
        "object_rotation_change_deg": total_rotation_change_deg,
        "end_effector_movement": total_ee_motion,
        "contact_steps": int(sum(contact_flags)),
        "max_object_step_displacement": float(object_step_displacements.max()) if object_step_displacements.size else 0.0,
        "final_success": bool(success),
        "post_probe_object_to_target_distance": None,
        "probe_triggered": False,
        "probe_start_step": None,
        "probe_end_step": None,
        "final_object_to_target_distance": final_target_distance,
        "target_instance": handles.get("target_instance"),
        "target_reference": handles.get("target_reference"),
        "target_body_names": handles.get("target_body_names", []),
        "target_site_names": handles.get("target_site_names", []),
        "video_path": str(video_path) if video_path else None,
        "probe_video_path": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline OpenVLA physics sweep for selected LIBERO goal tasks.")
    parser.add_argument("--output_dir", default="artifacts/baseline_goal_physics_sweep")
    parser.add_argument("--task_ids", default=None, help="Comma-separated task ids. Optional override.")
    parser.add_argument(
        "--task_names",
        default=None,
        help="Comma-separated task language strings or BDDL stems. Default: the five requested goal tasks.",
    )
    parser.add_argument("--frictions", default="0.05,0.2,0.5,0.7")
    parser.add_argument("--mass_scales", default="0.05,0.5,3.0,5.0,7.0,10.0")
    parser.add_argument("--episodes_per_setting", type=int, default=10)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_every", type=int, default=0)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    cfg = {
        "checkpoint": "openvla/openvla-7b-finetuned-libero-goal",
        "task_suite_name": "libero_goal",
        "center_crop": True,
        "num_trials_per_task": 1,
        "num_images_in_input": 1,
        "use_proprio": False,
        "use_l1_regression": False,
        "use_diffusion": False,
        "num_open_loop_steps": 1,
        "seed": 0,
        "paths": {
            "output_dir": str(Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval") / args.output_dir),
            "openvla_oft_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft",
            "libero_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO",
        },
        "policy": {
            "load_in_8bit": False,
            "load_in_4bit": False,
            "action_dim": 7,
        },
        "env": {
            "mode": "real",
        },
        "contact": {
            "use_filtered_contacts": True,
            "gripper_name_patterns": ["gripper", "finger", "leftpad", "rightpad"],
            "target_name_patterns": [],
            "min_contact_steps": 1,
        },
    }

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    video_dir = ensure_dir(output_dir / "videos")
    jsonl_path = output_dir / "episodes.jsonl"

    setup_real_paths(cfg)
    real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector = initialize_real_policy(cfg)
    real_cfg.num_open_loop_steps = 1
    task_suite = load_task_suite(real_cfg)
    task_ids = resolve_task_ids(
        task_suite,
        requested_specs=parse_csv_strings(args.task_names) or DEFAULT_SELECTED_TASK_SPECS,
        requested_ids=parse_csv_ints(args.task_ids),
    )
    resolved_tasks = [
        {
            "task_id": task_id,
            "language": getattr(task_suite.get_task(task_id), "language", ""),
            "bddl_stem": Path(str(getattr(task_suite.get_task(task_id), "bddl_file", ""))).stem,
        }
        for task_id in task_ids
    ]
    print({"selected_tasks": resolved_tasks}, flush=True)

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
            for setting_idx, setting in enumerate(settings):
                for episode_idx in range(int(args.episodes_per_setting)):
                    save_video = args.save_video and should_save_video(setting_idx, max(args.video_every, 1))
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
                        cfg=cfg,
                        max_steps=int(args.max_steps),
                        save_video=save_video,
                        video_dir=video_dir,
                        video_fps=int(args.video_fps),
                    )
                    rows.append(row)
                    append_jsonl(jsonl_path, row)
        finally:
            env.close()

    write_json(output_dir / "summary.json", summarize_rows(rows))


if __name__ == "__main__":
    main()
