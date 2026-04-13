"""
多GPU并行数据收集主脚本

用法:
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_collection.py --task PushCube-v1 --num-episodes 1000

每张GPU独立运行，收集数据到各自的目录
"""
import argparse
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
import gymnasium as gym
import numpy as np

from data_collection.collector import ProbeDataCollector
from data_collection.vla_policy import create_vla_policy


def collect_on_gpu(
    gpu_id: int,
    task_name: str,
    num_episodes_per_gpu: int,
    num_envs: int,
    policy_type: str,
    base_save_dir: str,
    **kwargs
):
    """
    单个GPU上的收集进程
    
    Args:
        gpu_id: GPU编号
        task_name: 任务名称
        num_episodes_per_gpu: 每个GPU收集的episode数
        num_envs: 每个GPU并行的环境数
        policy_type: VLA策略类型
        base_save_dir: 数据保存根目录
    """
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'  # 因为已经设置了CUDA_VISIBLE_DEVICES
    
    print(f"\n{'='*60}")
    print(f"GPU {gpu_id}: Starting data collection")
    print(f"  Task: {task_name}")
    print(f"  Episodes: {num_episodes_per_gpu}")
    print(f"  Parallel envs: {num_envs}")
    print(f"{'='*60}\n")
    
    # 创建保存目录
    save_dir = Path(base_save_dir) / f"gpu_{gpu_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载VLA策略（每个GPU一个）
    print(f"GPU {gpu_id}: Loading VLA policy...")
    policy = create_vla_policy(
        policy_type=policy_type,
        device=device,
        **kwargs.get('policy_kwargs', {})
    )
    policy.load()
    
    # 创建向量化环境
    print(f"GPU {gpu_id}: Creating {num_envs} parallel environments...")
    
    def make_env():
        """环境工厂函数"""
        env = gym.make(
            task_name,
            obs_mode="rgbd",  # RGB-D观测
            control_mode="pd_ee_delta_pose",  # 末端增量控制
            render_mode="rgb_array",
            # ManiSkill特定参数
            sim_backend="auto",  # 自动选择GPU物理引擎
        )
        return env
    
    # 创建向量化环境
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    
    # 为每个环境创建collector
    # 注意：这里简化处理，实际上vectorized env需要特殊处理
    # 暂时先用单环境版本，后面可以优化
    
    print(f"GPU {gpu_id}: Starting collection loop...")
    
    # 单环境收集（可以后续改成vectorized）
    env = make_env()
    collector = ProbeDataCollector(
        env=env,
        policy=policy,
        task_name=task_name,
        save_dir=save_dir,
        max_episode_steps=kwargs.get('max_episode_steps', 500),
        save_images=kwargs.get('save_images', True),
    )
    
    # 收集episodes
    successful_episodes = 0
    for ep_idx in range(num_episodes_per_gpu):
        episode_data = collector.collect_episode(ep_idx)
        
        if episode_data is not None and episode_data['outcome'] == 'success':
            successful_episodes += 1
        
        # 每50个episode打印一次进度
        if (ep_idx + 1) % 50 == 0:
            stats = collector.get_statistics()
            print(f"\nGPU {gpu_id} Progress: {ep_idx+1}/{num_episodes_per_gpu}")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            print(f"  Successful episodes: {successful_episodes}")
    
    # 最终统计
    final_stats = collector.get_statistics()
    print(f"\n{'='*60}")
    print(f"GPU {gpu_id}: Collection completed!")
    print(f"  Total episodes: {final_stats['total_episodes']}")
    print(f"  Successful: {final_stats['successful_episodes']}")
    print(f"  Success rate: {final_stats['success_rate']:.2%}")
    print(f"  Data saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Probe Data Collection")
    
    # 任务配置
    parser.add_argument('--task', type=str, required=True,
                        help='ManiSkill task name (e.g., PushCube-v1)')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Total number of episodes to collect')
    
    # GPU配置
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--num-envs-per-gpu', type=int, default=1,
                        help='Number of parallel environments per GPU')
    
    # VLA配置
    parser.add_argument('--policy', type=str, default='dummy',
                        choices=['openvla', 'dummy'],
                        help='VLA policy type (use dummy for testing)')
    parser.add_argument('--model-name', type=str, default='openvla/openvla-7b',
                        help='OpenVLA model name (if using openvla)')
    
    # 数据收集配置
    parser.add_argument('--save-dir', type=str, default='./data/probe_data',
                        help='Base directory to save collected data')
    parser.add_argument('--max-episode-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--no-save-images', action='store_true',
                        help='Do not save RGB images (to save disk space)')
    
    args = parser.parse_args()
    
    # 解析GPU列表
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    
    # 每个GPU收集的episodes
    episodes_per_gpu = args.num_episodes // num_gpus
    
    print(f"\n{'#'*70}")
    print(f"# Probe Data Collection - Multi-GPU")
    print(f"{'#'*70}")
    print(f"Task: {args.task}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"GPUs: {gpu_ids} ({num_gpus} GPUs)")
    print(f"Episodes per GPU: {episodes_per_gpu}")
    print(f"Parallel envs per GPU: {args.num_envs_per_gpu}")
    print(f"VLA policy: {args.policy}")
    print(f"Save directory: {args.save_dir}")
    print(f"{'#'*70}\n")
    
    # VLA策略参数
    policy_kwargs = {}
    if args.policy == 'openvla':
        policy_kwargs['model_name'] = args.model_name
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=collect_on_gpu,
            args=(
                gpu_id,
                args.task,
                episodes_per_gpu,
                args.num_envs_per_gpu,
                args.policy,
                args.save_dir,
            ),
            kwargs={
                'policy_kwargs': policy_kwargs,
                'max_episode_steps': args.max_episode_steps,
                'save_images': not args.no_save_images,
            }
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print(f"\n{'#'*70}")
    print(f"# All GPUs completed!")
    print(f"# Data saved to: {args.save_dir}")
    print(f"# Total collected: {args.num_episodes} episodes")
    print(f"{'#'*70}\n")
    
    # Merge统计信息
    merge_statistics(args.save_dir, gpu_ids)


def merge_statistics(base_dir: str, gpu_ids: list):
    """合并所有GPU的统计信息"""
    import json
    
    base_path = Path(base_dir)
    total_episodes = 0
    total_success = 0
    
    for gpu_id in gpu_ids:
        gpu_dir = base_path / f"gpu_{gpu_id}"
        if not gpu_dir.exists():
            continue
        
        # 统计这个GPU的数据
        hdf5_files = list(gpu_dir.glob("episode_*.hdf5"))
        gpu_episodes = len(hdf5_files)
        
        # 读取成功率（简化版本，实际需要读取HDF5）
        # 这里暂时跳过
        total_episodes += gpu_episodes
    
    stats = {
        'total_episodes': total_episodes,
        'num_gpus': len(gpu_ids),
        'gpu_ids': gpu_ids,
        'episodes_per_gpu': [len(list((base_path / f"gpu_{gid}").glob("episode_*.hdf5"))) 
                             for gid in gpu_ids if (base_path / f"gpu_{gid}").exists()],
    }
    
    stats_file = base_path / "collection_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {stats_file}")
    print(f"Total episodes collected: {total_episodes}")


if __name__ == '__main__':
    main()
