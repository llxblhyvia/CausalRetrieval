"""
数据验证和可视化工具
"""
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict
import json


def inspect_episode(hdf5_path: str):
    """检查单个episode的数据"""
    with h5py.File(hdf5_path, 'r') as f:
        print(f"\n{'='*60}")
        print(f"Episode: {hdf5_path}")
        print(f"{'='*60}")
        
        # 元数据
        print("\nMetadata:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")
        
        # Probe response
        print("\nProbe Response:")
        if 'probe_response' in f:
            for key in f['probe_response'].keys():
                val = f['probe_response'][key][()]
                print(f"  {key}: {val}")
        
        # Trajectory
        if 'trajectory' in f:
            num_steps = len(f['trajectory'].keys())
            print(f"\nTrajectory: {num_steps} steps")
        
        # Contact image
        if 'contact_image' in f:
            img_shape = f['contact_image'].shape
            print(f"\nContact Image: {img_shape}")


def visualize_probe_response(hdf5_path: str, save_path: str = None):
    """可视化probe response"""
    with h5py.File(hdf5_path, 'r') as f:
        response = {key: f['probe_response'][key][()] 
                   for key in f['probe_response'].keys()}
        
        outcome = f.attrs['outcome']
        episode_id = f.attrs['episode_id']
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Episode {episode_id} - Outcome: {outcome.upper()}", 
                 fontsize=14, fontweight='bold')
    
    # Force统计
    ax = axes[0, 0]
    force_metrics = ['mean_force', 'max_force', 'force_std']
    force_values = [response[k] for k in force_metrics]
    ax.bar(force_metrics, force_values, color='steelblue')
    ax.set_title('Force Metrics (N)')
    ax.set_ylabel('Force (N)')
    ax.grid(axis='y', alpha=0.3)
    
    # Motion统计
    ax = axes[0, 1]
    motion_metrics = ['total_ee_motion', 'total_obj_motion']
    motion_values = [response[k] for k in motion_metrics]
    ax.bar(motion_metrics, motion_values, color=['orange', 'green'])
    ax.set_title('Motion Metrics (m)')
    ax.set_ylabel('Distance (m)')
    ax.grid(axis='y', alpha=0.3)
    
    # Ratios
    ax = axes[1, 0]
    ratio_metrics = ['contact_ratio', 'motion_ratio']
    ratio_values = [response[k] for k in ratio_metrics]
    ax.bar(ratio_metrics, ratio_values, color=['purple', 'red'])
    ax.set_title('Ratio Metrics')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, max(ratio_values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # 总结表格
    ax = axes[1, 1]
    ax.axis('off')
    summary_data = [
        ['Metric', 'Value'],
        ['Probe Steps', f"{response['probe_steps']:.0f}"],
        ['Mean Force', f"{response['mean_force']:.3f} N"],
        ['Contact Ratio', f"{response['contact_ratio']:.2%}"],
        ['Motion Ratio', f"{response['motion_ratio']:.3f}"],
    ]
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def analyze_dataset(data_dir: str) -> Dict:
    """分析整个数据集"""
    data_path = Path(data_dir)
    
    # 收集所有episode文件
    all_episodes = []
    for gpu_dir in data_path.glob("gpu_*"):
        all_episodes.extend(list(gpu_dir.glob("episode_*.hdf5")))
    
    if len(all_episodes) == 0:
        print(f"No episodes found in {data_dir}")
        return {}
    
    print(f"\nAnalyzing {len(all_episodes)} episodes...")
    
    # 统计信息
    outcomes = {'success': 0, 'failure': 0}
    response_metrics = {
        'mean_force': [],
        'contact_ratio': [],
        'motion_ratio': [],
        'total_ee_motion': [],
        'total_obj_motion': [],
    }
    
    for ep_file in all_episodes:
        with h5py.File(ep_file, 'r') as f:
            outcome = f.attrs['outcome']
            outcomes[outcome] += 1
            
            for metric in response_metrics.keys():
                if metric in f['probe_response']:
                    val = f['probe_response'][metric][()]
                    response_metrics[metric].append(val)
    
    # 计算统计量
    stats = {
        'total_episodes': len(all_episodes),
        'outcomes': outcomes,
        'success_rate': outcomes['success'] / len(all_episodes),
    }
    
    for metric, values in response_metrics.items():
        values_arr = np.array(values)
        stats[metric] = {
            'mean': float(values_arr.mean()),
            'std': float(values_arr.std()),
            'min': float(values_arr.min()),
            'max': float(values_arr.max()),
        }
    
    return stats


def print_dataset_stats(stats: Dict):
    """打印数据集统计"""
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    
    print(f"\nTotal Episodes: {stats['total_episodes']}")
    print(f"  Success: {stats['outcomes']['success']}")
    print(f"  Failure: {stats['outcomes']['failure']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    print(f"\nProbe Response Metrics:")
    for metric in ['mean_force', 'contact_ratio', 'motion_ratio']:
        if metric in stats:
            m_stats = stats[metric]
            print(f"  {metric}:")
            print(f"    Mean: {m_stats['mean']:.4f} ± {m_stats['std']:.4f}")
            print(f"    Range: [{m_stats['min']:.4f}, {m_stats['max']:.4f}]")


def compare_success_failure(data_dir: str):
    """比较成功和失败episode的probe response差异"""
    data_path = Path(data_dir)
    
    success_responses = []
    failure_responses = []
    
    for gpu_dir in data_path.glob("gpu_*"):
        for ep_file in gpu_dir.glob("episode_*.hdf5"):
            with h5py.File(ep_file, 'r') as f:
                outcome = f.attrs['outcome']
                response = {key: f['probe_response'][key][()] 
                           for key in f['probe_response'].keys()}
                
                if outcome == 'success':
                    success_responses.append(response)
                else:
                    failure_responses.append(response)
    
    if not success_responses or not failure_responses:
        print("Need both success and failure examples")
        return
    
    # 对比分析
    print(f"\n{'='*60}")
    print("Success vs Failure Comparison")
    print(f"{'='*60}")
    print(f"Success episodes: {len(success_responses)}")
    print(f"Failure episodes: {len(failure_responses)}")
    
    metrics = ['mean_force', 'contact_ratio', 'motion_ratio']
    
    for metric in metrics:
        success_vals = np.array([r[metric] for r in success_responses])
        failure_vals = np.array([r[metric] for r in failure_responses])
        
        print(f"\n{metric}:")
        print(f"  Success: {success_vals.mean():.4f} ± {success_vals.std():.4f}")
        print(f"  Failure: {failure_vals.mean():.4f} ± {failure_vals.std():.4f}")
        print(f"  Difference: {success_vals.mean() - failure_vals.mean():.4f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/data_inspector.py <data_dir>  # Analyze dataset")
        print("  python tools/data_inspector.py <episode.hdf5>  # Inspect single episode")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if path.endswith('.hdf5'):
        # 单个episode
        inspect_episode(path)
        visualize_probe_response(path)
    else:
        # 整个数据集
        stats = analyze_dataset(path)
        print_dataset_stats(stats)
        compare_success_failure(path)
