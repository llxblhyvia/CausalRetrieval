"""
Data collection script for CausalRetrieval.

Collects interaction episodes from ManiSkill3 manipulation environments.
Each episode stores:
  - Pre-manipulation RGB image (I)
  - Object segmentation mask (M)
  - Action trajectory {a_1, ..., a_T}
  - Force/contact trajectory {f_1, ..., f_T}
  - Post-manipulation RGB image (I')
  - Binary outcome label (y: success/failure)
  - Failure type label (c: e.g. drop, timeout, collision)
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs


# ---------------------------------------------------------------------------
# Task registry: maps task aliases to ManiSkill env IDs & task-specific info
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "pick_cube": {
        "env_id": "PickCube-v1",
        "target_obj_name": "cube",
    },
    "push_cube": {
        "env_id": "PushCube-v1",
        "target_obj_name": "cube",
    },
    "stack_cube": {
        "env_id": "StackCube-v1",
        "target_obj_name": "cubeA",
    },
    "peg_insertion": {
        "env_id": "PegInsertionSide-v1",
        "target_obj_name": "peg",
    },
    "lift_peg": {
        "env_id": "LiftPegUpright-v1",
        "target_obj_name": "peg",
    },
    "plug_charger": {
        "env_id": "PlugCharger-v1",
        "target_obj_name": "charger",
    },
    "turn_faucet": {
        "env_id": "TurnFaucet-v1",
        "target_obj_name": "target_link",
    },
    "pick_ycb": {
        "env_id": "PickSingleYCB-v1",
        "target_obj_name": "obj",
    },
}


# ---------------------------------------------------------------------------
# Action strategies
# ---------------------------------------------------------------------------


def random_action(env, obs, env_idx=0):
    """Sample a uniformly random action from the action space."""
    return env.action_space.sample()


def noisy_zero_action(env, obs, env_idx=0):
    """Small Gaussian noise around zero – mainly useful for data diversity."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action += np.random.normal(0, 0.1, size=action.shape).astype(np.float32)
    return np.clip(action, env.action_space.low, env.action_space.high)


ACTION_STRATEGIES = {
    "random": random_action,
    "noisy_zero": noisy_zero_action,
}


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# Failure type codes
FAILURE_TYPES = {
    "success": 0,
    "drop": 1,
    "timeout": 2,
    "collision": 3,
    "unknown_failure": 4,
}


def classify_failure(
    success: bool,
    truncated: bool,
    obj_z_trajectory: np.ndarray,
    contact_force_norms: np.ndarray,
    table_height: float = 0.0,
    drop_z_thresh: float = -0.02,
    collision_force_thresh: float = 50.0,
) -> tuple[int, str]:
    """
    Programmatically assign a failure label based on episode statistics.

    Returns (failure_code, failure_name).
    """
    if success:
        return FAILURE_TYPES["success"], "success"

    # Drop: object fell below table surface
    if obj_z_trajectory.size > 0 and np.any(
        obj_z_trajectory < table_height + drop_z_thresh
    ):
        return FAILURE_TYPES["drop"], "drop"

    # Collision: abnormally high contact forces detected
    if contact_force_norms.size > 0 and np.any(
        contact_force_norms > collision_force_thresh
    ):
        return FAILURE_TYPES["collision"], "collision"

    # Timeout: episode ended by truncation without success
    if truncated:
        return FAILURE_TYPES["timeout"], "timeout"

    return FAILURE_TYPES["unknown_failure"], "unknown_failure"


# ---------------------------------------------------------------------------
# Image / mask extraction helpers
# ---------------------------------------------------------------------------


def extract_rgb(obs: dict, camera_name: str = "base_camera") -> np.ndarray:
    """
    Extract an RGB image (H, W, 3) uint8 from the observation dict.
    Works with obs_mode='rgbd' or 'rgb+depth+segmentation'.
    """
    sensor_data = obs["sensor_data"]
    cam_data = sensor_data[camera_name]
    rgb = cam_data["rgb"]  # (num_envs, H, W, 4) or (H, W, 4) torch tensor
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    # Remove batch dim if present
    if rgb.ndim == 4:
        rgb = rgb[0]
    return rgb[..., :3].astype(np.uint8)


def extract_segmentation_mask(
    obs: dict, target_obj_id: int, camera_name: str = "base_camera"
) -> np.ndarray:
    """
    Extract a binary segmentation mask (H, W) for the target object.
    Requires obs_mode that includes 'segmentation'.
    """
    sensor_data = obs["sensor_data"]
    cam_data = sensor_data[camera_name]
    seg = cam_data["segmentation"]  # (num_envs, H, W, 1)
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    if seg.ndim == 4:
        seg = seg[0]
    mask = (seg[..., 0] == target_obj_id).astype(np.uint8)
    return mask


def get_target_object_seg_id(env, target_obj_name: str) -> int:
    """
    Resolve the per-actor segmentation ID used in the rendered segmentation map.
    ManiSkill assigns each actor a unique integer segmentation id.
    """
    # Try looking up the actor by name in the scene
    # In ManiSkill3, actors have a per_scene_id attribute
    target = None
    for actor in env.unwrapped.scene.actors.values():
        if hasattr(actor, "name") and target_obj_name in actor.name:
            target = actor
            break
    if target is None:
        print(
            f"[WARN] Could not find actor '{target_obj_name}' in scene; "
            "using seg_id=1 as fallback."
        )
        return 1
    # The segmentation id used in rendering is the per_scene_id + 1
    # (0 is reserved for background)
    if hasattr(target, "per_scene_id"):
        return int(target.per_scene_id) + 1
    return 1


def get_object_position(env, target_obj_name: str) -> np.ndarray:
    """Get the current 3D position of the target object."""
    for actor in env.unwrapped.scene.actors.values():
        if hasattr(actor, "name") and target_obj_name in actor.name:
            pos = actor.pose.p
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            if pos.ndim == 2:
                pos = pos[0]
            return pos
    return np.zeros(3)


# ---------------------------------------------------------------------------
# Contact force extraction
# ---------------------------------------------------------------------------


def get_gripper_contact_forces(env) -> np.ndarray:
    """
    Get net contact force on the robot's TCP / gripper links.
    Returns a 3D force vector (x, y, z).
    """
    try:
        agent = env.unwrapped.agent
        # Panda has finger links: left/right
        tcp_link = agent.tcp
        forces = tcp_link.get_net_contact_forces()
        if isinstance(forces, torch.Tensor):
            forces = forces.cpu().numpy()
        if forces.ndim == 2:
            forces = forces[0]
        return forces
    except Exception:
        return np.zeros(3)


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------


def collect_episode(env, action_fn, target_obj_name: str, max_steps: int):
    """
    Run one episode, collecting the full experience tuple.

    Returns a dict with all fields needed for the memory bank, or None if
    the episode is degenerate (e.g. zero-length).
    """
    obs, info = env.reset()

    # Resolve target object segmentation ID
    target_seg_id = get_target_object_seg_id(env, target_obj_name)

    # Pre-manipulation snapshot
    pre_rgb = extract_rgb(obs)
    seg_mask = extract_segmentation_mask(obs, target_seg_id)

    # Trajectory buffers
    actions_list = []
    forces_list = []
    obj_z_list = []

    terminated = False
    truncated = False
    success = False
    total_reward = 0.0

    for step in range(max_steps):
        action = action_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Record action
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        actions_list.append(action.copy())

        # Record contact forces on gripper
        contact_force = get_gripper_contact_forces(env)
        forces_list.append(contact_force.copy())

        # Track object z for drop detection
        obj_pos = get_object_position(env, target_obj_name)
        obj_z_list.append(obj_pos[2])

        total_reward += float(reward) if np.isscalar(reward) else float(reward.item())

        if terminated or truncated:
            break

    if len(actions_list) == 0:
        return None

    # Post-manipulation snapshot
    post_rgb = extract_rgb(obs)

    # Determine success from the last info dict
    if "success" in info:
        s = info["success"]
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        success = bool(np.any(s))
    else:
        success = False

    # Build trajectories
    action_traj = np.stack(actions_list, axis=0)  # (T, action_dim)
    force_traj = np.stack(forces_list, axis=0)  # (T, 3)
    obj_z_traj = np.array(obj_z_list)  # (T,)
    contact_force_norms = np.linalg.norm(force_traj, axis=-1)  # (T,)

    # Classify failure type
    failure_code, failure_name = classify_failure(
        success=success,
        truncated=truncated,
        obj_z_trajectory=obj_z_traj,
        contact_force_norms=contact_force_norms,
    )

    return {
        "pre_rgb": pre_rgb,  # (H, W, 3) uint8
        "seg_mask": seg_mask,  # (H, W) uint8
        "action_traj": action_traj,  # (T, action_dim) float32
        "force_traj": force_traj,  # (T, 3) float32
        "post_rgb": post_rgb,  # (H, W, 3) uint8
        "success": success,  # bool
        "failure_code": failure_code,  # int
        "failure_name": failure_name,  # str
        "num_steps": len(actions_list),
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
# HDF5 storage
# ---------------------------------------------------------------------------


def save_episodes_to_hdf5(episodes: list[dict], path: str, task_name: str):
    """
    Save a list of collected episodes to an HDF5 file.

    File layout:
        /meta/task_name
        /meta/num_episodes
        /episode_0000/pre_rgb          (H, W, 3)  uint8
        /episode_0000/seg_mask          (H, W)     uint8
        /episode_0000/action_traj       (T, D)     float32
        /episode_0000/force_traj        (T, 3)     float32
        /episode_0000/post_rgb          (H, W, 3)  uint8
        /episode_0000/success           scalar     bool
        /episode_0000/failure_code      scalar     int
        /episode_0000/failure_name      scalar     str
        /episode_0000/num_steps         scalar     int
        /episode_0000/total_reward      scalar     float
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["task_name"] = task_name
        meta.attrs["num_episodes"] = len(episodes)

        for idx, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{idx:04d}")
            grp.create_dataset(
                "pre_rgb", data=ep["pre_rgb"], compression="gzip", compression_opts=4
            )
            grp.create_dataset(
                "seg_mask", data=ep["seg_mask"], compression="gzip", compression_opts=4
            )
            grp.create_dataset("action_traj", data=ep["action_traj"])
            grp.create_dataset("force_traj", data=ep["force_traj"])
            grp.create_dataset(
                "post_rgb", data=ep["post_rgb"], compression="gzip", compression_opts=4
            )
            grp.attrs["success"] = ep["success"]
            grp.attrs["failure_code"] = ep["failure_code"]
            grp.attrs["failure_name"] = ep["failure_name"]
            grp.attrs["num_steps"] = ep["num_steps"]
            grp.attrs["total_reward"] = ep["total_reward"]

    print(f"Saved {len(episodes)} episodes to {path}")


# ---------------------------------------------------------------------------
# Contrastive pair construction helpers
# ---------------------------------------------------------------------------


def build_contrastive_pairs(episodes: list[dict]) -> list[dict]:
    """
    Construct positive and negative pairs for contrastive training.

    Positive pair: same failure_code (shared object-action-effect structure).
    Negative pair: different failure_code.

    Returns a list of dicts with keys: idx_a, idx_b, label (1=pos, 0=neg).
    """
    from collections import defaultdict

    by_failure = defaultdict(list)
    for i, ep in enumerate(episodes):
        by_failure[ep["failure_code"]].append(i)

    pairs = []
    codes = list(by_failure.keys())

    # Positive pairs: sample within the same failure group
    for code in codes:
        indices = by_failure[code]
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, min(i + 5, len(indices))):
                pairs.append(
                    {"idx_a": indices[i], "idx_b": indices[j], "label": 1}
                )

    # Negative pairs: sample across different failure groups
    for ci in range(len(codes)):
        for cj in range(ci + 1, len(codes)):
            idxs_a = by_failure[codes[ci]]
            idxs_b = by_failure[codes[cj]]
            n_neg = min(len(idxs_a), len(idxs_b), 10)
            for k in range(n_neg):
                pairs.append(
                    {"idx_a": idxs_a[k % len(idxs_a)],
                     "idx_b": idxs_b[k % len(idxs_b)],
                     "label": 0}
                )

    return pairs


def save_pairs_to_hdf5(pairs: list[dict], path: str):
    """Save contrastive pairs to a separate HDF5 file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        n = len(pairs)
        f.create_dataset("idx_a", data=np.array([p["idx_a"] for p in pairs]))
        f.create_dataset("idx_b", data=np.array([p["idx_b"] for p in pairs]))
        f.create_dataset("label", data=np.array([p["label"] for p in pairs]))
        f.attrs["num_pairs"] = n
    print(f"Saved {len(pairs)} contrastive pairs to {path}")


# ---------------------------------------------------------------------------
# Physical property variation (for contrastive data)
# ---------------------------------------------------------------------------


def make_env(
    env_id: str,
    obs_mode: str = "rgb+depth+segmentation",
    control_mode: str = "pd_ee_delta_pose",
    render_mode: str = "rgb_array",
    max_episode_steps: int | None = None,
    sim_backend: str = "auto",
    num_envs: int = 1,
    **kwargs,
) -> gym.Env:
    """
    Create a ManiSkill3 environment with the given configuration.
    """
    make_kwargs = dict(
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        num_envs=num_envs,
    )
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps
    make_kwargs.update(kwargs)
    return gym.make(env_id, **make_kwargs)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect_data(
    task: str,
    num_episodes: int,
    output_dir: str,
    action_strategy: str = "random",
    max_steps: int = 50,
    num_envs: int = 1,
    seed: int = 0,
    save_pairs: bool = True,
):
    """
    Main entry point: collect episodes and save to disk.
    """
    task_info = TASK_REGISTRY[task]
    env_id = task_info["env_id"]
    target_obj_name = task_info["target_obj_name"]

    print(f"Task: {task} (env_id={env_id})")
    print(f"Collecting {num_episodes} episodes with strategy='{action_strategy}'")
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_id, num_envs=num_envs, max_episode_steps=max_steps)
    action_fn = ACTION_STRATEGIES[action_strategy]

    episodes = []
    success_count = 0

    for ep_idx in range(num_episodes):
        ep = collect_episode(env, action_fn, target_obj_name, max_steps)
        if ep is None:
            print(f"  Episode {ep_idx}: degenerate, skipping")
            continue

        episodes.append(ep)
        success_count += int(ep["success"])

        if (ep_idx + 1) % 10 == 0 or ep_idx == num_episodes - 1:
            print(
                f"  Episode {ep_idx + 1}/{num_episodes} | "
                f"success_rate={success_count}/{len(episodes)} "
                f"({100 * success_count / max(len(episodes), 1):.1f}%) | "
                f"last_failure={ep['failure_name']}"
            )

    env.close()

    # Save episodes
    out_path = os.path.join(output_dir, f"{task}_episodes.h5")
    save_episodes_to_hdf5(episodes, out_path, task)

    # Build and save contrastive pairs
    if save_pairs and len(episodes) > 1:
        pairs = build_contrastive_pairs(episodes)
        pairs_path = os.path.join(output_dir, f"{task}_pairs.h5")
        save_pairs_to_hdf5(pairs, pairs_path)

    # Print summary
    failure_dist = {}
    for ep in episodes:
        name = ep["failure_name"]
        failure_dist[name] = failure_dist.get(name, 0) + 1
    print("\n--- Collection Summary ---")
    print(f"Total episodes: {len(episodes)}")
    print(f"Success rate: {100 * success_count / max(len(episodes), 1):.1f}%")
    print(f"Failure distribution: {failure_dist}")
    print(f"Data saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CausalRetrieval: Collect manipulation episodes from ManiSkill3"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick_cube",
        choices=list(TASK_REGISTRY.keys()),
        help="Task to collect data for",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save collected data",
    )
    parser.add_argument(
        "--action-strategy",
        type=str,
        default="random",
        choices=list(ACTION_STRATEGIES.keys()),
        help="Action generation strategy",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (use >1 for GPU sim)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--no-pairs",
        action="store_true",
        help="Skip contrastive pair construction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_data(
        task=args.task,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        action_strategy=args.action_strategy,
        max_steps=args.max_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        save_pairs=not args.no_pairs,
    )
