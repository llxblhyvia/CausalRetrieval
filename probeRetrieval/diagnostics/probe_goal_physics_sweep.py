from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

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
from rollout.video_utils import FrameBuffer, should_save_video, write_video


DEFAULT_SELECTED_TASK_SPECS = [
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_wine_bottle_on_top_of_the_cabinet",
]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def parse_csv_strings(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def sanitize_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def task_spec_tokens(task: Any) -> set[str]:
    tokens = set()
    language = getattr(task, "language", None)
    if language:
        tokens.add(sanitize_name(language))
    bddl_file = getattr(task, "bddl_file", None)
    if bddl_file:
        tokens.add(sanitize_name(Path(str(bddl_file)).stem))
    return tokens


def resolve_task_ids(task_suite: Any, requested_specs: Sequence[str] | None, requested_ids: Sequence[int] | None) -> list[int]:
    if requested_ids:
        return [int(task_id) for task_id in requested_ids]

    requested_specs = list(requested_specs or DEFAULT_SELECTED_TASK_SPECS)
    normalized_specs = [sanitize_name(spec) for spec in requested_specs]
    resolved: list[int] = []
    unresolved = set(normalized_specs)
    for task_id in range(int(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        tokens = task_spec_tokens(task)
        if any(spec in tokens for spec in normalized_specs):
            resolved.append(task_id)
            unresolved.difference_update(tokens.intersection(unresolved))
    if unresolved:
        available = []
        for task_id in range(int(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            available.append(
                {
                    "task_id": task_id,
                    "language": getattr(task, "language", ""),
                    "bddl_stem": Path(str(getattr(task, "bddl_file", ""))).stem,
                }
            )
        raise ValueError(f"Could not resolve requested task specs: {sorted(unresolved)}. Available tasks: {available}")
    return resolved


def model_name(model: Any, kind: str, idx: int) -> str:
    getter = getattr(model, f"{kind}_id2name", None)
    if getter is not None:
        return str(getter(idx) or "")
    try:
        return str(model.id2name(idx, kind) or "")
    except TypeError:
        return str(model.id2name(kind, idx) or "")


def names_by_kind(model: Any, kind: str, count: int) -> list[str]:
    return [model_name(model, kind, idx) for idx in range(count)]


def matching_indices(names: Sequence[str], patterns: Sequence[str]) -> list[int]:
    out = []
    for idx, name in enumerate(names):
        normalized = sanitize_name(name)
        if any(pattern and pattern in normalized for pattern in patterns):
            out.append(idx)
    return out


def get_task_bddl_path(task: Any) -> Path:
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def parse_obj_of_interest(task: Any) -> tuple[str | None, str | None]:
    text = get_task_bddl_path(task).read_text(encoding="utf-8")
    match = re.search(r"\(:obj_of_interest\s+(.*?)\)", text, re.DOTALL)
    if not match:
        return None, None
    names = [token for token in re.findall(r"([a-zA-Z0-9_]+)", match.group(1)) if token != "obj_of_interest"]
    if not names:
        return None, None
    return names[0], names[1] if len(names) > 1 else None


def target_patterns(name: str | None) -> list[str]:
    if not name:
        return []
    patterns = {sanitize_name(name)}
    root = sanitize_name(re.sub(r"_\d+$", "", name))
    patterns.add(root)
    if root == "salad_dressing":
        patterns.add("new_salad_dressing")
    return sorted(patterns)


def resolve_task_handles(env: Any, task: Any) -> Dict[str, Any]:
    target_instance, target_reference = parse_obj_of_interest(task)
    model = env.sim.model
    body_names = names_by_kind(model, "body", int(model.nbody))
    geom_names = names_by_kind(model, "geom", int(model.ngeom))
    site_names = names_by_kind(model, "site", int(model.nsite))
    body_ids = matching_indices(body_names, target_patterns(target_instance))
    geom_ids = matching_indices(geom_names, target_patterns(target_instance))
    if not body_ids and geom_ids:
        body_ids = sorted({int(model.geom_bodyid[idx]) for idx in geom_ids})
    if not geom_ids and body_ids:
        body_id_set = set(body_ids)
        geom_ids = [idx for idx in range(int(model.ngeom)) if int(model.geom_bodyid[idx]) in body_id_set]
    target_body_id = body_ids[0] if body_ids else None
    site_ids = matching_indices(site_names, target_patterns(target_reference))
    target_site_id = site_ids[0] if site_ids else None
    reference_body_ids = matching_indices(body_names, target_patterns(target_reference))
    reference_body_id = reference_body_ids[0] if reference_body_ids else None
    return {
        "target_instance": target_instance,
        "target_reference": target_reference,
        "target_body_id": target_body_id,
        "target_site_id": target_site_id,
        "target_reference_body_id": reference_body_id,
        "target_body_names": [body_names[idx] for idx in body_ids],
        "target_geom_names": [geom_names[idx] for idx in geom_ids],
        "target_site_names": [site_names[idx] for idx in site_ids],
        "geom_ids": geom_ids,
        "body_ids": body_ids,
    }


def capture_physics_snapshot(env: Any) -> Dict[str, np.ndarray]:
    model = env.sim.model
    return {
        "geom_friction": np.asarray(model.geom_friction, dtype=np.float64).copy(),
        "body_mass": np.asarray(model.body_mass, dtype=np.float64).copy(),
    }


def restore_physics_snapshot(env: Any, snapshot: Mapping[str, np.ndarray]) -> None:
    model = env.sim.model
    model.geom_friction[:] = snapshot["geom_friction"]
    model.body_mass[:] = snapshot["body_mass"]
    if hasattr(env.sim, "forward"):
        env.sim.forward()


def apply_physics_modifiers(
    env: Any,
    handles: Mapping[str, Any],
    *,
    friction_value: float | None,
    mass_scale: float | None,
) -> Dict[str, Any]:
    model = env.sim.model
    original_frictions = []
    original_masses = []
    for geom_id in handles.get("geom_ids", []):
        original_frictions.append(float(model.geom_friction[geom_id, 0]))
    for body_id in handles.get("body_ids", []):
        original_masses.append(float(model.body_mass[body_id]))
    if friction_value is not None:
        for geom_id in handles.get("geom_ids", []):
            model.geom_friction[geom_id, 0] = float(friction_value)
    if mass_scale is not None:
        for body_id in handles.get("body_ids", []):
            model.body_mass[body_id] = float(model.body_mass[body_id] * mass_scale)
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    return {
        "friction_value": float(friction_value) if friction_value is not None else None,
        "mass_scale": float(mass_scale) if mass_scale is not None else None,
        "original_friction_values": original_frictions,
        "original_body_masses": original_masses,
    }


def body_position(env: Any, body_id: int | None) -> np.ndarray:
    if body_id is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float32).copy()


def body_quaternion(env: Any, body_id: int | None) -> np.ndarray:
    if body_id is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float32).copy()


def site_position(env: Any, site_id: int | None) -> np.ndarray:
    if site_id is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float32).copy()


def target_position(env: Any, handles: Mapping[str, Any]) -> np.ndarray:
    site_id = handles.get("target_site_id")
    if site_id is not None:
        return site_position(env, int(site_id))
    ref_body_id = handles.get("target_reference_body_id")
    if ref_body_id is not None:
        return body_position(env, int(ref_body_id))
    return np.zeros(3, dtype=np.float32)


def get_end_effector_position(obs: Mapping[str, Any]) -> np.ndarray:
    for key in ("robot0_eef_pos", "eef_pos", "ee_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1)[:3]
    return np.zeros(3, dtype=np.float32)


def extract_agentview(obs: Mapping[str, Any]) -> np.ndarray | None:
    image = obs.get("agentview_image")
    if image is None:
        return None
    return np.asarray(image)[::-1, ::-1]


def translational_force_norm(env: Any, body_id: int | None) -> float:
    if body_id is None:
        return 0.0
    cfrc_ext = getattr(getattr(env.sim, "data", None), "cfrc_ext", None)
    if cfrc_ext is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(cfrc_ext[body_id], dtype=np.float32)[:3]))


def quat_geodesic_degrees(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.linalg.norm(q0) == 0 or np.linalg.norm(q1) == 0:
        return 0.0
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def mean_std_max(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), float(arr.max())


def make_detector_cfg(base_cfg: Mapping[str, Any], target_instance: str | None) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    env_cfg = dict(cfg.get("env", {}))
    contact_cfg = dict(cfg.get("contact", {}))
    target_patterns = list(contact_cfg.get("target_name_patterns", []))
    for pattern in target_patterns_from_instance(target_instance):
        if pattern not in target_patterns:
            target_patterns.append(pattern)
    contact_cfg["target_name_patterns"] = target_patterns
    env_cfg["target_object_name"] = target_instance
    cfg["env"] = env_cfg
    cfg["contact"] = contact_cfg
    return cfg


def target_patterns_from_instance(target_instance: str | None) -> list[str]:
    if not target_instance:
        return []
    root = sanitize_name(re.sub(r"_\d+$", "", target_instance))
    patterns = [sanitize_name(target_instance)]
    if root and root not in patterns:
        patterns.append(root)
    return patterns


def contact_active(detector: ContactDetector, env: Any) -> bool:
    sim = getattr(env, "sim", getattr(getattr(env, "env", None), "sim", None))
    data = getattr(sim, "data", None)
    ncon = int(getattr(data, "ncon", 0) or 0)
    matched, details = detector._filtered_match(sim, data, ncon)
    if detector.use_filtered_contacts and details.get("filter_available"):
        return bool(matched)
    return ncon > 0


def run_probe(
    env: Any,
    probe_cfg: Mapping[str, Any],
    handles: Mapping[str, Any],
    detector: ContactDetector,
    obs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Dict[str, Any], list[np.ndarray]]:
    from rollout.real_libero import real_probe_actions

    object_positions = [body_position(env, handles.get("target_body_id"))]
    object_quats = [body_quaternion(env, handles.get("target_body_id"))]
    ee_positions = [get_end_effector_position(obs)]
    target_pos = target_position(env, handles)
    contact_steps = 0
    done = False
    success = False
    force_values: list[float] = []
    probe_contact_flags: list[bool] = []
    frames: list[np.ndarray] = []

    for action in real_probe_actions(probe_cfg):
        obs, reward, step_done, info = env.step(action.tolist())
        current_object_pos = body_position(env, handles.get("target_body_id"))
        object_positions.append(current_object_pos)
        object_quats.append(body_quaternion(env, handles.get("target_body_id")))
        ee_positions.append(get_end_effector_position(obs))
        force_values.append(translational_force_norm(env, handles.get("target_body_id")))
        current_contact = contact_active(detector, env)
        probe_contact_flags.append(bool(current_contact))
        if current_contact:
            contact_steps += 1
        frame = extract_agentview(obs)
        if frame is not None:
            frames.append(frame)
        done = bool(step_done)
        success = bool(info.get("success")) or reward > 0.5
        if done:
            break

    object_positions_arr = np.stack(object_positions, axis=0)
    ee_positions_arr = np.stack(ee_positions, axis=0)
    step_displacements = np.linalg.norm(np.diff(object_positions_arr, axis=0), axis=1) if len(object_positions_arr) > 1 else np.zeros(0)
    force_mean, force_std, force_max = mean_std_max(force_values)
    metrics = {
        "probe_num_steps": len(object_positions) - 1,
        "probe_obj_displacement": float(np.linalg.norm(object_positions_arr[-1] - object_positions_arr[0])),
        "probe_total_obj_motion": float(step_displacements.sum()) if step_displacements.size else 0.0,
        "probe_max_object_step_displacement": float(step_displacements.max()) if step_displacements.size else 0.0,
        "probe_object_rotation_change_deg": quat_geodesic_degrees(object_quats[0], object_quats[-1]),
        "probe_ee_motion": float(np.linalg.norm(np.diff(ee_positions_arr, axis=0), axis=1).sum()) if len(ee_positions_arr) > 1 else 0.0,
        "probe_contact_steps": int(contact_steps),
        "probe_contact_ratio": float(contact_steps / max(len(object_positions) - 1, 1)),
        "post_probe_object_to_target_distance": float(np.linalg.norm(object_positions_arr[-1] - target_pos)),
        "probe_mean_force": force_mean,
        "probe_max_force": force_max,
        "probe_force_std": force_std,
        "probe_force_values": force_values,
        "probe_contact_flags": probe_contact_flags,
    }
    return obs, {"done": done, "success": success, **metrics}, frames


def build_setting_rows(
    friction_values: Sequence[float],
    mass_scales: Sequence[float],
) -> list[Dict[str, Any]]:
    rows = []
    for friction_value in friction_values:
        rows.append({"sweep_type": "friction", "friction_value": float(friction_value), "mass_scale": 1.0})
    for mass_scale in mass_scales:
        rows.append({"sweep_type": "mass", "friction_value": None, "mass_scale": float(mass_scale)})
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "by_task": {}}
    if not rows:
        return summary
    summary["overall"] = {
        "num_episodes": len(rows),
        "success_rate": float(np.mean([bool(row["final_success"]) for row in rows])),
    }
    task_names = sorted({str(row["task_name"]) for row in rows})
    for task_name in task_names:
        task_rows = [row for row in rows if row["task_name"] == task_name]
        task_summary: Dict[str, Any] = {}
        for setting_name in sorted({str(row["setting_name"]) for row in task_rows}):
            setting_rows = [row for row in task_rows if row["setting_name"] == setting_name]
            post_probe_distances = [
                float(row["post_probe_object_to_target_distance"])
                for row in setting_rows
                if row["post_probe_object_to_target_distance"] is not None
            ]
            task_summary[setting_name] = {
                "n": len(setting_rows),
                "success_rate": float(np.mean([bool(row["final_success"]) for row in setting_rows])),
                "mean_force": float(np.mean([float(row["mean_force"]) for row in setting_rows])),
                "mean_contact_ratio": float(np.mean([float(row["contact_ratio"]) for row in setting_rows])),
                "mean_probe_obj_displacement": float(np.mean([float(row["probe_obj_displacement"]) for row in setting_rows])),
                "mean_post_probe_object_to_target_distance": (
                    float(np.mean(post_probe_distances)) if post_probe_distances else None
                ),
            }
        summary["by_task"][task_name] = task_summary
    return summary


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
    pre_probe = FrameBuffer(maxlen=max(video_fps, 1))
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
    }
    probe_frames: list[np.ndarray] = []
    success = False

    try:
        for step in range(int(max_steps)):
            frame = extract_agentview(obs)
            if save_video and frame is not None:
                frames.append(frame)
                pre_probe.append(frame)

            is_contact = contact_active(detector, env)
            contact_flags.append(bool(is_contact))
            force_values.append(translational_force_norm(env, handles.get("target_body_id")))
            event = detector.check(env, step)

            if event.triggered and not probe_triggered:
                probe_triggered = True
                probe_start_step = step
                obs, probe_result, probe_frames = run_probe(env, cfg, handles, detector, obs)
                probe_end_step = step + int(probe_result["probe_num_steps"])
                probe_metrics.update(probe_result)
                ee_positions.append(get_end_effector_position(obs))
                object_positions.append(body_position(env, handles.get("target_body_id")))
                object_quats.append(body_quaternion(env, handles.get("target_body_id")))
                contact_flags.extend([bool(flag) for flag in probe_result["probe_contact_flags"]])
                force_values.extend([float(value) for value in probe_result["probe_force_values"]])
                if probe_result["done"]:
                    success = bool(probe_result["success"])
                    break

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
    video_path = None
    probe_video_path = None
    if save_video and frames:
        safe_task = sanitize_name(task_description)[:80]
        stem = (
            f"task{task_id:02d}_ep{episode_idx:03d}_{sanitize_name(setting['setting_name'])}_{safe_task}"
        )
        video_path = write_video(video_dir / f"{stem}.mp4", frames, fps=video_fps)
        if probe_frames:
            probe_video_path = write_video(video_dir / f"{stem}_probe.mp4", list(pre_probe.to_list()) + probe_frames, fps=video_fps)

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
        "post_probe_object_to_target_distance": float(probe_metrics["post_probe_object_to_target_distance"]),
        "probe_triggered": bool(probe_triggered),
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "final_object_to_target_distance": final_target_distance,
        "target_instance": handles.get("target_instance"),
        "target_reference": handles.get("target_reference"),
        "target_body_names": handles.get("target_body_names", []),
        "target_site_names": handles.get("target_site_names", []),
        "video_path": str(video_path) if video_path else None,
        "probe_video_path": str(probe_video_path) if probe_video_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe-enabled physics sweep for selected LIBERO goal tasks.")
    parser.add_argument("--output_dir", default="artifacts/probe_goal_physics_sweep")
    parser.add_argument("--task_ids", default=None, help="Comma-separated task ids. Optional override.")
    parser.add_argument(
        "--task_names",
        default=None,
        help="Comma-separated task language strings or BDDL stems. Default: the five requested goal tasks.",
    )
    parser.add_argument("--frictions", default="0.2,0.7")
    parser.add_argument("--mass_scales", default="0.5,3.0,7.0")
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
        "probe": {
            "num_close_steps": 2,
            "num_lift_steps": 3,
            "num_hold_steps": 1,
            "lift_delta_z": 0.015,
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
        requested_specs=parse_csv_strings(args.task_names),
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
