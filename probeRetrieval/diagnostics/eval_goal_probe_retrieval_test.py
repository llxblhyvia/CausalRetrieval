from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from probe.contact_detector import ContactDetector
from retrieval.image_embedder import create_image_embedder
from retrieval.image_retrieval import retrieve_top_k
from retrieval.memory_bank import MemoryBank
from retrieval.probe_rerank import (
    aggregate_actions_major_vote,
    aggregate_actions_soft,
    rerank_by_response,
)
from retrieval.response_bank import ResponseBank
from retrieval.response_features import RESPONSE_FEATURE_KEYS
from rollout.real_libero import (
    get_initial_state,
    get_real_action_chunk,
    initialize_real_policy,
    load_task_suite,
    make_env_for_task,
    setup_real_paths,
)
from rollout.rollout_utils import append_jsonl, ensure_dir, write_json
from rollout.video_utils import FrameBuffer, extend_frames, should_save_video, write_video
from vla.openvla_policy import fuse_actions

from diagnostics.probe_goal_physics_sweep import (
    DEFAULT_SELECTED_TASK_SPECS,
    apply_physics_modifiers,
    build_setting_rows,
    make_detector_cfg,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    resolve_task_handles,
    resolve_task_ids,
    restore_physics_snapshot,
    capture_physics_snapshot,
    run_probe,
)


TEST_FRICTIONS = "0.05,0.5"
TEST_MASS_SCALES = "0.05,5.0,10.0"
VARIANTS = ("full_probe_retrieval", "probe_only", "random_retrieve", "image_only_retrieval")


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "by_task": {}}
    if not rows:
        return summary
    summary["overall"] = {
        "num_episodes": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
    }
    for task_name in sorted({str(row["task_name"]) for row in rows}):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        per_setting: Dict[str, Any] = {}
        for setting_name in sorted({str(row["setting_name"]) for row in task_rows}):
            setting_rows = [row for row in task_rows if row["setting_name"] == setting_name]
            per_setting[setting_name] = {
                "n": len(setting_rows),
                "success_rate": float(np.mean([bool(row["success"]) for row in setting_rows])),
                "mean_num_ranked": float(np.mean([float(row["num_ranked_candidates"]) for row in setting_rows])),
                "mean_num_successful_top5": float(np.mean([float(row["num_successful_top5"]) for row in setting_rows])),
            }
        summary["by_task"][task_name] = per_setting
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
    embedder: Any,
    bank: MemoryBank,
    response_bank: ResponseBank,
    cfg: Mapping[str, Any],
    max_steps: int,
    save_video: bool,
    video_dir: Path,
    video_fps: int,
    variant: str,
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
    apply_physics_modifiers(
        env,
        handles,
        friction_value=setting.get("friction_value"),
        mass_scale=setting.get("mass_scale"),
    )

    allowed_indices = [idx for idx, item in enumerate(bank.items) if item.task_name == task_description]
    action_queue: deque[np.ndarray] = deque(maxlen=real_cfg.num_open_loop_steps)
    contact_triggered = False
    probe_start_step = None
    probe_end_step = None
    query_features: Dict[str, float] = {}
    ranked: list[dict[str, Any]] = []
    retrieved_ids: list[str] = []
    fused_norms: list[float] = []
    fusion_start_step = None
    action_r = None
    success = False
    full_frames: list[np.ndarray] = []
    pre_probe_frames = FrameBuffer(maxlen=max(video_fps, 1))
    probe_clip_frames: list[np.ndarray] = []

    try:
        for step in range(int(max_steps)):
            observation, img = prepare_observation(obs, resize_size)
            if save_video:
                extend_frames(full_frames, [img])
                pre_probe_frames.append(img)
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
            action = action_v
            event = detector.check(env, step)
            if not contact_triggered and event.triggered:
                contact_triggered = True
                probe_start_step = step
                top = retrieve_top_k(
                    embedder.embed(img),
                    bank,
                    int(cfg.get("retrieval", {}).get("top_k", 20)),
                    eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                    allowed_indices=allowed_indices,
                )
                if variant != "image_only_retrieval":
                    obs, probe_result, probe_frames = run_probe(env, cfg, handles, detector, obs)
                    probe_end_step = step + int(probe_result["probe_num_steps"])
                    if save_video:
                        extend_frames(probe_clip_frames, probe_frames)
                        extend_frames(full_frames, probe_frames)
                    action_queue.clear()
                    query_features = {
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
                    if probe_result["done"]:
                        success = bool(probe_result["success"])
                        break

                if variant == "full_probe_retrieval":
                    ranked = rerank_by_response(
                        query_features,
                        top,
                        response_bank,
                        task_description,
                        top_k=int(cfg.get("retrieval", {}).get("rerank_top_k", 5)),
                        feature_keys=cfg.get("retrieval", {}).get("response_feature_keys", RESPONSE_FEATURE_KEYS),
                        feature_weights=cfg.get("retrieval", {}).get("response_feature_weights"),
                        eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                    )
                elif variant == "random_retrieve":
                    pool = [
                        dict(candidate, response_success=bool(response_bank.get(str(candidate.get("episode_id", ""))).final_success))
                        for candidate in top
                        if response_bank.get(str(candidate.get("episode_id", ""))) is not None
                    ]
                    rng = random.Random((task_id + 1) * 100000 + episode_idx * 100 + step)
                    rng.shuffle(pool)
                    ranked = pool[: int(cfg.get("retrieval", {}).get("rerank_top_k", 5))]
                    for local_idx, candidate in enumerate(ranked):
                        candidate["response_distance"] = float(local_idx)
                elif variant == "image_only_retrieval":
                    ranked = []
                    for candidate in top[: int(cfg.get("retrieval", {}).get("rerank_top_k", 5))]:
                        response_item = response_bank.get(str(candidate.get("episode_id", "")))
                        out = dict(candidate)
                        out["response_success"] = bool(response_item.final_success) if response_item is not None else bool(
                            bank.items[int(candidate["index"])].success
                        )
                        out["response_distance"] = float(1.0 - float(candidate.get("score", 0.0)))
                        ranked.append(out)

                if variant in ("full_probe_retrieval", "random_retrieve", "image_only_retrieval"):
                    if str(cfg.get("retrieval", {}).get("aggregation", "soft")).lower() == "vote":
                        action_r, retrieved_ids = aggregate_actions_major_vote(
                            ranked,
                            bank,
                            response_bank=response_bank,
                            successful_only=True,
                        )
                    else:
                        action_r, retrieved_ids = aggregate_actions_soft(
                            ranked,
                            bank,
                            response_bank=response_bank,
                            successful_only=True,
                            temperature=float(cfg.get("retrieval", {}).get("temperature", 0.1)),
                            eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                        )
                fusion_start_step = step
                continue

            if action_r is not None and fusion_start_step is not None:
                if step - fusion_start_step < int(cfg.get("fusion", {}).get("max_steps", 30)):
                    action = fuse_actions(
                        action_v,
                        action_r,
                        float(cfg.get("fusion", {}).get("alpha", 0.5)),
                        int(cfg.get("policy", {}).get("action_dim", 7)),
                    )
                    fused_norms.append(float(np.linalg.norm(action - action_v)))
            obs, reward, done, info = env.step(action.tolist())
            if save_video:
                frame = obs.get("agentview_image")
                if frame is not None:
                    extend_frames(full_frames, [frame[::-1, ::-1]])
            if done or bool(info.get("success")) or reward > 0.5:
                success = True
                break
    finally:
        restore_physics_snapshot(env, snapshot)

    num_successful_top5 = 0
    for candidate in ranked:
        if bool(candidate.get("response_success", False)):
            num_successful_top5 += 1

    safe_task = task_description.replace(" ", "_")[:80]
    stem = f"{variant}_task{task_id:02d}_ep{episode_idx:03d}_{setting['setting_name']}_{safe_task}"
    video_path = None
    probe_video_path = None
    if save_video and full_frames:
        video_path = write_video(video_dir / f"{stem}.mp4", full_frames, fps=video_fps)
        if probe_clip_frames:
            probe_video_path = write_video(
                video_dir / f"{stem}_probe.mp4",
                pre_probe_frames.to_list() + probe_clip_frames,
                fps=video_fps,
            )

    return {
        "episode_id": f"task_{task_id:02d}/trial_{episode_idx:04d}/{setting['setting_name']}",
        "variant": variant,
        "task_name": task_description,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "setting_name": str(setting["setting_name"]),
        "sweep_type": str(setting["sweep_type"]),
        "friction_value": setting.get("friction_value"),
        "mass_scale": setting.get("mass_scale"),
        "contact_triggered": bool(contact_triggered),
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "query_features": query_features,
        "retrieved_candidate_ids": retrieved_ids,
        "retrieved_candidates": ranked,
        "num_ranked_candidates": len(ranked),
        "num_successful_top5": int(num_successful_top5),
        "fused_action_stats": {
            "count": len(fused_norms),
            "mean_delta_norm": float(np.mean(fused_norms)) if fused_norms else 0.0,
            "max_delta_norm": float(np.max(fused_norms)) if fused_norms else 0.0,
        },
        "video_path": str(video_path) if video_path else None,
        "probe_video_path": str(probe_video_path) if probe_video_path else None,
        "success": bool(success),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probe retrieval on LIBERO-goal physics test settings.")
    parser.add_argument("--output_dir", default="artifacts/goal_probe_retrieval_test_eval")
    parser.add_argument("--memory_dir", default="artifacts/real_collect/memory_bank")
    parser.add_argument(
        "--response_jsonl",
        default="artifacts/collectingPhase_w_probe_sweep_5tasks/episodes.jsonl",
    )
    parser.add_argument("--task_ids", default=None)
    parser.add_argument("--task_names", default=None)
    parser.add_argument("--frictions", default=TEST_FRICTIONS)
    parser.add_argument("--mass_scales", default=TEST_MASS_SCALES)
    parser.add_argument("--episodes_per_setting", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_every", type=int, default=0)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--variant", choices=VARIANTS, default="full_probe_retrieval")
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
            "memory_dir": str(Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval") / args.memory_dir),
            "response_jsonl": str(Path("/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval") / args.response_jsonl),
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
        "retrieval": {
            "top_k": 20,
            "rerank_top_k": 5,
            "response_feature_keys": RESPONSE_FEATURE_KEYS,
            "response_feature_weights": None,
            "aggregation": "soft",
            "temperature": 0.1,
            "eps": 1.0e-6,
        },
        "fusion": {"alpha": 0.5, "max_steps": 30},
    }

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    log_path = output_dir / "probe_retrieval_test_episodes.jsonl"
    video_dir = ensure_dir(output_dir / "videos")

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
    bank = MemoryBank.load(cfg["paths"]["memory_dir"])
    response_bank = ResponseBank.load_jsonl(cfg["paths"]["response_jsonl"])

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
                    save_video = bool(args.save_video) and should_save_video(episode_idx, max(int(args.video_every), 1))
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
                        bank=bank,
                        response_bank=response_bank,
                        cfg=cfg,
                        max_steps=int(args.max_steps),
                        save_video=save_video,
                        video_dir=video_dir,
                        video_fps=int(args.video_fps),
                        variant=str(args.variant),
                    )
                    rows.append(row)
                    append_jsonl(log_path, row)
        finally:
            env.close()

    summary = summarize_rows(rows)
    write_json(output_dir / "probe_retrieval_test_summary.json", summary)


if __name__ == "__main__":
    main()
