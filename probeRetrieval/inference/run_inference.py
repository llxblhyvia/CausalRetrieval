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
from retrieval.image_retrieval import retrieve_top_k
from retrieval.memory_bank import MemoryBank
from retrieval.probe_rerank import aggregate_retrieved_action, rerank_by_probe
from rollout.rollout_utils import add_common_args, append_jsonl, config_from_args, ensure_dir, seed_everything, write_json
from vla.openvla_policy import create_policy, fuse_actions


VARIANTS = ("baseline_vla", "image_retrieval_only", "full_probe_rerank")


def run_episode(task_name: str, trial_idx: int, variant: str, cfg: Mapping[str, Any], bank: MemoryBank | None) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0)) + 10000 + trial_idx
    env = create_env(task_name, cfg, seed=seed)
    policy = create_policy(cfg)
    detector = ContactDetector(cfg)
    probe = ProbeRunner(cfg)
    embedder = create_image_embedder(cfg)
    obs = env.reset()
    detector.reset()
    contact_triggered = False
    probe_features: Dict[str, float] = {}
    retrieved_candidate_ids: list[str] = []
    fused_norms = []
    probe_start_step = None
    probe_end_step = None
    action_r = None
    ranked = []
    success = False
    fusion_start_step = None

    for step in range(int(cfg.get("max_steps", 220))):
        action_v = policy.predict(obs, task_name)
        action = action_v
        if variant != "baseline_vla":
            event = detector.check(env, step)
            if event.triggered and not contact_triggered:
                contact_triggered = True
                probe_start_step = step
                image = extract_image(obs, cfg)
                top = retrieve_top_k(
                    embedder.embed(image),
                    bank,
                    int(cfg.get("retrieval", {}).get("top_k", 10)),
                    eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                ) if bank is not None else []
                if variant == "full_probe_rerank":
                    trace = probe.run(env, obs, action_v)
                    probe_features = extract_probe_features(trace, cfg)
                    probe_end_step = step + len(trace.actions)
                    obs = trace.observations[-1]
                    ranked = rerank_by_probe(
                        probe_features,
                        top,
                        bank,
                        int(cfg.get("retrieval", {}).get("rerank_top_k", 5)),
                        image_weight=float(cfg.get("retrieval", {}).get("image_weight", 0.35)),
                        probe_weight=float(cfg.get("retrieval", {}).get("probe_weight", 0.65)),
                        eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                    ) if bank is not None else []
                else:
                    ranked = top
                action_r, retrieved_candidate_ids = aggregate_retrieved_action(
                    ranked,
                    bank,
                    successful_only=bool(cfg.get("retrieval", {}).get("successful_only", True)),
                    eps=float(cfg.get("retrieval", {}).get("eps", 1.0e-6)),
                ) if bank is not None else (None, [])
                fusion_start_step = step
            if action_r is not None:
                max_fusion_steps = int(cfg.get("fusion", {}).get("max_steps", 30))
                if fusion_start_step is None or step - fusion_start_step < max_fusion_steps:
                    action = fuse_actions(
                        action_v,
                        action_r,
                        alpha=float(cfg.get("fusion", {}).get("alpha", 0.5)),
                        action_dim=int(cfg.get("policy", {}).get("action_dim", 7)),
                    )
                    fused_norms.append(float(np.linalg.norm(action - action_v)))
        result = env.step(action)
        obs = result.obs
        success = check_success(result.info, result.reward)
        if result.done:
            break
    env.close()
    return {
        "episode_id": f"{variant}/{task_name}/trial_{trial_idx:04d}",
        "variant": variant,
        "task_name": task_name,
        "seed": seed,
        "contact_triggered": contact_triggered,
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "probe_features": probe_features,
        "retrieved_candidate_ids": retrieved_candidate_ids,
        "retrieved_candidates": ranked,
        "fused_action_stats": {
            "count": len(fused_norms),
            "mean_delta_norm": float(np.mean(fused_norms)) if fused_norms else 0.0,
            "max_delta_norm": float(np.max(fused_norms)) if fused_norms else 0.0,
        },
        "success": bool(success),
    }


def run_inference(cfg: Mapping[str, Any], variants: list[str]) -> Dict[str, Any]:
    if cfg.get("env", {}).get("mode") == "real" or cfg.get("policy", {}).get("mode") == "openvla":
        from rollout.real_libero import run_real_inference

        return run_real_inference(cfg, variants)
    seed_everything(int(cfg.get("seed", 0)))
    output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "artifacts"))
    memory_dir = cfg.get("paths", {}).get("memory_dir", output_dir / "memory_bank")
    bank = None
    if any(v != "baseline_vla" for v in variants):
        bank = MemoryBank.load(memory_dir)
    all_rows = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
        log_path = output_dir / f"{variant}_episodes.jsonl"
        for task_name in get_libero_object_tasks():
            for trial_idx in range(int(cfg.get("num_trials_per_task", 1))):
                row = run_episode(task_name, trial_idx, variant, cfg, bank)
                all_rows.append(row)
                append_jsonl(log_path, row)
    summary = summarize_rows(all_rows)
    write_json(output_dir / "inference_summary.json", summary)
    return summary


def summarize_rows(rows: list[Mapping[str, Any]]) -> Dict[str, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIBERO object retrieval variants.")
    add_common_args(parser, default_config="infer_libero_object.yaml")
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="Variant to run. Repeatable.")
    args, unknown = parser.parse_known_args()
    cfg = config_from_args(args, unknown)
    variants = args.variant or list(VARIANTS)
    summary = run_inference(cfg, variants)
    print(summary)


if __name__ == "__main__":
    main()
