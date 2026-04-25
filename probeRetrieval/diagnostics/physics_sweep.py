from __future__ import annotations

import argparse
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from rollout.real_libero import (
    get_initial_state,
    get_real_action_chunk,
    initialize_real_policy,
    load_task_suite,
    make_env_for_task,
    setup_real_paths,
)
from rollout.rollout_utils import append_jsonl, ensure_dir, normalize_action, write_json
from rollout.video_utils import FrameBuffer, should_save_video, write_video


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def sanitize_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


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
    target = names[0]
    basket = names[1] if len(names) > 1 else None
    return target, basket


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


def target_patterns(target_instance: str | None) -> list[str]:
    if not target_instance:
        return []
    patterns = {sanitize_name(target_instance)}
    object_type = sanitize_name(re.sub(r"_\d+$", "", target_instance))
    patterns.add(object_type)
    if object_type == "salad_dressing":
        patterns.add("new_salad_dressing")
    return sorted(patterns)


def resolve_target_handles(env: Any, task: Any) -> Dict[str, Any]:
    target_instance, basket_instance = parse_obj_of_interest(task)
    model = env.sim.model
    body_names = names_by_kind(model, "body", int(model.nbody))
    geom_names = names_by_kind(model, "geom", int(model.ngeom))
    patterns = target_patterns(target_instance)
    basket_patterns = target_patterns(basket_instance)
    body_ids = matching_indices(body_names, patterns)
    geom_ids = matching_indices(geom_names, patterns)
    basket_body_ids = matching_indices(body_names, basket_patterns)
    if not body_ids and geom_ids:
        body_ids = sorted({int(model.geom_bodyid[idx]) for idx in geom_ids})
    if not geom_ids and body_ids:
        geom_ids = [idx for idx in range(int(model.ngeom)) if int(model.geom_bodyid[idx]) in set(body_ids)]
    target_body_id = body_ids[0] if body_ids else None
    basket_body_id = basket_body_ids[0] if basket_body_ids else None
    joint_ids: list[int] = []
    if body_ids:
        body_id_set = set(body_ids)
        for joint_id in range(int(model.njnt)):
            if int(model.jnt_bodyid[joint_id]) in body_id_set:
                joint_ids.append(joint_id)
    return {
        "target_instance": target_instance,
        "basket_instance": basket_instance,
        "body_ids": body_ids,
        "geom_ids": geom_ids,
        "joint_ids": joint_ids,
        "target_body_id": target_body_id,
        "basket_body_id": basket_body_id,
        "body_names": [body_names[idx] for idx in body_ids],
        "geom_names": [geom_names[idx] for idx in geom_ids],
    }


def capture_physics_snapshot(env: Any, handles: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    model = env.sim.model
    joint_ids = list(handles.get("joint_ids", []))
    snapshots = {
        "geom_friction": np.asarray(model.geom_friction, dtype=np.float64).copy(),
        "body_mass": np.asarray(model.body_mass, dtype=np.float64).copy(),
    }
    if joint_ids:
        snapshots["dof_damping"] = np.asarray(model.dof_damping, dtype=np.float64).copy()
    return snapshots


def restore_physics_snapshot(env: Any, snapshot: Mapping[str, np.ndarray]) -> None:
    model = env.sim.model
    model.geom_friction[:] = snapshot["geom_friction"]
    model.body_mass[:] = snapshot["body_mass"]
    if "dof_damping" in snapshot:
        model.dof_damping[:] = snapshot["dof_damping"]
    if hasattr(env.sim, "forward"):
        env.sim.forward()


def apply_physics_modifiers(
    env: Any,
    handles: Mapping[str, Any],
    friction_value: float,
    mass_scale: float,
    damping_scale: float = 1.0,
) -> Dict[str, Any]:
    model = env.sim.model
    applied: Dict[str, Any] = {
        "friction_value": friction_value,
        "mass_scale": mass_scale,
        "damping_scale": damping_scale,
        "geom_ids": list(handles.get("geom_ids", [])),
        "body_ids": list(handles.get("body_ids", [])),
        "joint_ids": list(handles.get("joint_ids", [])),
    }
    for geom_id in handles.get("geom_ids", []):
        model.geom_friction[geom_id, 0] = float(friction_value)
    for body_id in handles.get("body_ids", []):
        model.body_mass[body_id] = float(model.body_mass[body_id] * mass_scale)
    for joint_id in handles.get("joint_ids", []):
        dof_adr = int(model.jnt_dofadr[joint_id])
        if dof_adr >= 0:
            model.dof_damping[dof_adr] = float(model.dof_damping[dof_adr] * damping_scale)
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    return applied


def body_position(env: Any, body_id: int | None) -> np.ndarray:
    if body_id is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float32).copy()


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


def classify_failure(
    success: bool,
    friction_value: float,
    mass_scale: float,
    ee_motion: float,
    object_disp: float,
    max_object_step_disp: float,
    final_obj_to_basket: float,
) -> str:
    if success:
        return "success"
    if friction_value <= 0.1 and object_disp > 0.025 and final_obj_to_basket > 0.08:
        return "slip_low_friction"
    if friction_value >= 2.0 and ee_motion > 0.12 and object_disp < 0.02:
        return "move_not_enough_high_friction"
    if mass_scale >= 5.0 and ee_motion > 0.12 and object_disp < 0.025:
        return "inertia_high_mass"
    if mass_scale <= 0.5 and max_object_step_disp > 0.03:
        return "overshoot_low_mass"
    return "other_failure"


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"tasks": {}}
    for task_name in sorted({str(row["task_name"]) for row in rows}):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        combo_summary = {}
        for friction in sorted({float(row["friction_value"]) for row in task_rows}):
            for mass_scale in sorted({float(row["mass_scale"]) for row in task_rows}):
                combo_rows = [
                    row
                    for row in task_rows
                    if float(row["friction_value"]) == friction and float(row["mass_scale"]) == mass_scale
                ]
                if not combo_rows:
                    continue
                combo_key = f"mu={friction:g},m={mass_scale:g}"
                failure_counts: Dict[str, int] = {}
                for row in combo_rows:
                    failure_counts[str(row["failure_mode"])] = failure_counts.get(str(row["failure_mode"]), 0) + 1
                combo_summary[combo_key] = {
                    "n": len(combo_rows),
                    "success_rate": float(np.mean([bool(row["success"]) for row in combo_rows])),
                    "mean_object_displacement": float(np.mean([float(row["object_displacement"]) for row in combo_rows])),
                    "mean_ee_motion": float(np.mean([float(row["ee_motion"]) for row in combo_rows])),
                    "failure_counts": dict(sorted(failure_counts.items())),
                }
        summary["tasks"][task_name] = combo_summary
    return summary


def run_single_episode(
    env: Any,
    task_suite: Any,
    task_id: int,
    task_description: str,
    episode_idx: int,
    combo_idx: int,
    real_cfg: Any,
    model: Any,
    resize_size: int,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    noisy_action_projector: Any,
    friction_value: float,
    mass_scale: float,
    damping_scale: float,
    save_video: bool,
    video_dir: Path,
    video_fps: int,
    max_steps: int,
    progress_every: int,
) -> Dict[str, Any]:
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    env.reset()
    obs = env.set_init_state(get_initial_state(real_cfg, task_suite, task_id, episode_idx))
    for _ in range(real_cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(real_cfg.model_family))

    handles = resolve_target_handles(env, task_suite.get_task(task_id))
    snapshot = capture_physics_snapshot(env, handles)
    applied = apply_physics_modifiers(env, handles, friction_value, mass_scale, damping_scale)

    action_queue: deque[np.ndarray] = deque(maxlen=real_cfg.num_open_loop_steps)
    frames: list[np.ndarray] = []
    pre_contact = FrameBuffer(maxlen=max(video_fps, 1))
    success = False
    contact_step = None
    ee_positions = [get_end_effector_position(obs)]
    object_positions = [body_position(env, handles.get("target_body_id"))]
    initial_object_pos = object_positions[0].copy()
    basket_pos = body_position(env, handles.get("basket_body_id"))
    step_object_displacements = []
    for step in range(int(max_steps)):
        frame = extract_agentview(obs)
        if save_video and frame is not None:
            frames.append(frame)
            pre_contact.append(frame)
        if len(action_queue) == 0:
            chunk_start = time.time()
            print(
                f"[physics_sweep] query_chunk task={task_id} ep={episode_idx} combo={combo_idx} "
                f"step={step} mu={friction_value:g} m={mass_scale:g}",
                flush=True,
            )
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
            print(
                f"[physics_sweep] got_chunk task={task_id} ep={episode_idx} combo={combo_idx} "
                f"step={step} dt={time.time() - chunk_start:.2f}s",
                flush=True,
            )
        action = normalize_action(np.asarray(action_queue.popleft(), dtype=np.float32), 7)
        obs, reward, done, info = env.step(action.tolist())
        ee_positions.append(get_end_effector_position(obs))
        current_object_pos = body_position(env, handles.get("target_body_id"))
        step_object_displacements.append(float(np.linalg.norm(current_object_pos - object_positions[-1])))
        object_positions.append(current_object_pos)
        if contact_step is None and int(getattr(getattr(env.sim, "data", None), "ncon", 0) or 0) > 0:
            contact_step = step
        if done or bool(info.get("success")) or reward > 0.5:
            success = True
            break
        if progress_every > 0 and (step + 1) % progress_every == 0:
            print(
                f"[physics_sweep] task={task_id} ep={episode_idx} combo={combo_idx} "
                f"step={step + 1} mu={friction_value:g} m={mass_scale:g}",
                flush=True,
            )
    restore_physics_snapshot(env, snapshot)

    ee_positions_arr = np.stack(ee_positions, axis=0)
    object_positions_arr = np.stack(object_positions, axis=0)
    ee_motion = float(np.linalg.norm(np.diff(ee_positions_arr, axis=0), axis=1).sum()) if len(ee_positions_arr) > 1 else 0.0
    object_disp = float(np.linalg.norm(object_positions_arr[-1] - initial_object_pos))
    final_obj_to_basket = float(np.linalg.norm(object_positions_arr[-1][:2] - basket_pos[:2])) if basket_pos.size else 0.0
    failure_mode = classify_failure(
        success=success,
        friction_value=friction_value,
        mass_scale=mass_scale,
        ee_motion=ee_motion,
        object_disp=object_disp,
        max_object_step_disp=max(step_object_displacements or [0.0]),
        final_obj_to_basket=final_obj_to_basket,
    )

    video_path = None
    if save_video and frames:
        safe_task = sanitize_name(task_description)[:80]
        video_path = write_video(
            video_dir / f"task{task_id:02d}_ep{episode_idx:02d}_combo{combo_idx:03d}_{safe_task}.mp4",
            frames,
            fps=video_fps,
        )
        if contact_step is not None:
            clip_frames = pre_contact.to_list() + frames[max(contact_step, 0) : min(contact_step + video_fps, len(frames))]
            write_video(
                video_dir / f"task{task_id:02d}_ep{episode_idx:02d}_combo{combo_idx:03d}_{safe_task}_contact.mp4",
                clip_frames,
                fps=video_fps,
            )

    return {
        "task_id": task_id,
        "task_name": task_description,
        "episode_idx": episode_idx,
        "combo_idx": combo_idx,
        "friction_value": friction_value,
        "mass_scale": mass_scale,
        "damping_scale": damping_scale,
        "success": bool(success),
        "failure_mode": failure_mode,
        "contact_step": contact_step,
        "object_displacement": object_disp,
        "max_object_step_displacement": float(max(step_object_displacements or [0.0])),
        "ee_motion": ee_motion,
        "initial_object_pos": initial_object_pos.tolist(),
        "final_object_pos": object_positions_arr[-1].tolist(),
        "basket_pos": basket_pos.tolist(),
        "final_obj_to_basket_xy": final_obj_to_basket,
        "target_instance": handles.get("target_instance"),
        "target_body_names": handles.get("body_names", []),
        "target_geom_names": handles.get("geom_names", []),
        "applied": applied,
        "video_path": str(video_path) if video_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small OpenVLA physics sweep on LIBERO object tasks.")
    parser.add_argument("--output_dir", default="artifacts/physics_sweep")
    parser.add_argument("--frictions", default="0.05,0.1,0.2,0.5,1.0,2.0,4.0")
    parser.add_argument("--mass_scales", default="0.2,0.5,1.0,2.0,5.0,10.0")
    parser.add_argument("--damping_scale", type=float, default=1.0)
    parser.add_argument("--episodes_per_combo", type=int, default=10)
    parser.add_argument("--task_ids", default=None, help="Comma-separated task ids. Default: all 10 libero_object tasks.")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_every", type=int, default=0, help="Save one video every N combos per task.")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=280)
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()

    cfg = {
        "checkpoint": "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-7b-oft",
        "task_suite_name": "libero_object",
        "center_crop": True,
        "num_trials_per_task": 1,
        "seed": 0,
        "paths": {
            "output_dir": str(Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval") / args.output_dir),
            "openvla_oft_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft",
            "libero_repo": "/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO",
        },
        "policy": {
            "load_in_8bit": False,
            "load_in_4bit": False,
        },
    }

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    videos_dir = ensure_dir(output_dir / "videos")
    jsonl_path = output_dir / "episodes.jsonl"

    setup_real_paths(cfg)
    real_cfg, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector = initialize_real_policy(cfg)
    task_suite = load_task_suite(real_cfg)

    task_ids = parse_csv_ints(args.task_ids) or list(range(task_suite.n_tasks))
    frictions = parse_csv_floats(args.frictions)
    mass_scales = parse_csv_floats(args.mass_scales)
    rows = []
    combo_idx = 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env, task_description = make_env_for_task(task, real_cfg)
        try:
            for friction_value in frictions:
                for mass_scale in mass_scales:
                    for episode_idx in range(args.episodes_per_combo):
                        save_video = args.save_video and should_save_video(combo_idx, max(args.video_every, 1))
                        print(
                            f"[physics_sweep] start task={task_id} ep={episode_idx} combo={combo_idx} "
                            f"mu={friction_value:g} m={mass_scale:g}",
                            flush=True,
                        )
                        row = run_single_episode(
                            env=env,
                            task_suite=task_suite,
                            task_id=task_id,
                            task_description=task_description,
                            episode_idx=episode_idx,
                            combo_idx=combo_idx,
                            real_cfg=real_cfg,
                            model=model,
                            resize_size=resize_size,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            friction_value=friction_value,
                            mass_scale=mass_scale,
                            damping_scale=float(args.damping_scale),
                            save_video=save_video,
                            video_dir=videos_dir,
                            video_fps=int(args.video_fps),
                            max_steps=int(args.max_steps),
                            progress_every=int(args.progress_every),
                        )
                        rows.append(row)
                        append_jsonl(jsonl_path, row)
                        print(
                            f"[physics_sweep] done task={task_id} ep={episode_idx} combo={combo_idx} "
                            f"success={row['success']} mode={row['failure_mode']} "
                            f"obj_disp={row['object_displacement']:.4f}",
                            flush=True,
                        )
                    combo_idx += 1
        finally:
            env.close()

    summary = summarize_rows(rows)
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
