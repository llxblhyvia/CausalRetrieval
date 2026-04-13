"""
核心数据收集逻辑：
1. 使用VLA接近目标
2. 检测contact
3. 执行probe并记录response
4. 继续VLA策略至episode结束
"""
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import h5py
from datetime import datetime

from configs.probe_actions import get_probe_config, ProbeActionConfig
from utils.contact_detection import ContactDetector, ProbeResponseRecorder
from data_collection.vla_policy import OpenVLAPolicy


class ProbeDataCollector:
    """单个环境的数据收集器"""
    
    def __init__(
        self,
        env: gym.Env,
        policy: OpenVLAPolicy,
        task_name: str,
        save_dir: str,
        max_episode_steps: int = 500,
        save_images: bool = True,
    ):
        """
        Args:
            env: ManiSkill环境
            policy: VLA策略
            task_name: 任务名称
            save_dir: 数据保存目录
            max_episode_steps: 单个episode最大步数
            save_images: 是否保存RGB图像
        """
        self.env = env
        self.policy = policy
        self.task_name = task_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_episode_steps = max_episode_steps
        self.save_images = save_images
        
        # 获取probe配置
        self.probe_config = get_probe_config(task_name)
        
        # Contact检测器
        self.contact_detector = ContactDetector(
            force_threshold=self.probe_config.force_threshold,
            min_contact_steps=self.probe_config.contact_duration,
        )
        
        # Response记录器
        self.response_recorder = ProbeResponseRecorder()
        
        # 统计信息
        self.episode_count = 0
        self.success_count = 0
    
    def collect_episode(self, episode_idx: int) -> Optional[Dict]:
        """
        收集单个episode的数据
        
        Returns:
            episode_data = {
                'episode_id': int,
                'task_name': str,
                'contact_image': np.ndarray,  # contact时刻的图像
                'planned_action': np.ndarray,  # VLA计划的action_v
                'probe_response': Dict,  # 8个response指标
                'trajectory': List[Dict],  # 完整轨迹
                'outcome': str,  # 'success' or 'failure'
                'total_steps': int,
            }
        """
        print(f"\n=== Episode {episode_idx} ===")
        
        # 1. Reset环境
        obs, info = self.env.reset()
        self.contact_detector.reset()
        self.response_recorder.reset()
        
        episode_data = {
            'episode_id': episode_idx,
            'task_name': self.task_name,
            'timestamp': datetime.now().isoformat(),
        }
        
        trajectory = []
        contact_detected = False
        contact_image = None
        planned_action_v = None
        probe_response = None
        
        # 2. Phase 1: VLA接近物体直到contact
        print("Phase 1: Approaching target...")
        for step in range(self.max_episode_steps):
            # VLA预测动作
            action = self._get_vla_action(obs)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 检测contact
            contacts = self._extract_contacts(info)
            is_contact, contact_info = self.contact_detector.detect_contact(obs, contacts)
            
            trajectory.append({
                'obs': self._extract_obs_dict(obs),
                'action': action,
                'reward': reward,
                'is_contact': is_contact,
            })
            
            if is_contact:
                print(f"  ✓ Contact detected at step {step}")
                contact_detected = True
                contact_image = self._extract_image(obs)
                planned_action_v = action  # 记录VLA计划做的动作
                
                # 3. Phase 2: 执行probe
                print("Phase 2: Executing probe...")
                probe_response = self._execute_probe(next_obs, contact_info)
                
                # 4. Phase 3: 继续VLA策略至结束
                print("Phase 3: Continuing main policy...")
                final_outcome = self._continue_vla_policy(
                    next_obs, 
                    trajectory, 
                    remaining_steps=self.max_episode_steps - step - 1
                )
                
                break
            
            obs = next_obs
            
            if terminated or truncated:
                print(f"  Episode ended before contact (step {step})")
                break
        
        # 5. 整理episode数据
        if not contact_detected:
            print("  ✗ No contact detected, discarding episode")
            return None
        
        episode_data.update({
            'contact_image': contact_image,
            'planned_action': planned_action_v,
            'probe_response': probe_response,
            'trajectory': trajectory,
            'outcome': final_outcome,
            'total_steps': len(trajectory),
        })
        
        # 6. 保存数据
        self._save_episode(episode_data)
        
        # 更新统计
        self.episode_count += 1
        if final_outcome == 'success':
            self.success_count += 1
        
        print(f"  Outcome: {final_outcome}")
        print(f"  Success rate: {self.success_count}/{self.episode_count} = {self.success_count/self.episode_count:.2%}")
        
        return episode_data
    
    def _execute_probe(self, obs: Dict, contact_info: Dict) -> Dict:
        """
        执行probe并记录response
        
        Args:
            obs: contact时刻的观测
            contact_info: contact检测器返回的接触信息
        
        Returns:
            probe_response: 8个指标的字典
        """
        self.response_recorder.reset()
        
        # 获取初始状态
        ee_pose = self._get_ee_pose(obs)
        obj_pose = self._get_object_pose(obs)
        
        # 生成probe动作
        probe_action = self._generate_probe_action(ee_pose, contact_info)
        
        # 执行probe（持续多步）
        for probe_step in range(self.probe_config.duration_steps):
            # 执行probe动作
            next_obs, _, _, _, info = self.env.step(probe_action)
            
            # 记录response数据
            tcp_force = self._extract_tcp_force(next_obs)
            force_mag = np.linalg.norm(tcp_force)
            
            contacts = self._extract_contacts(info)
            is_contact = len(contacts) > 0
            
            new_ee_pose = self._get_ee_pose(next_obs)
            new_obj_pose = self._get_object_pose(next_obs)
            
            self.response_recorder.record_step(
                force_magnitude=force_mag,
                is_contact=is_contact,
                ee_pos=new_ee_pose[:3],  # 只要位置
                obj_pos=new_obj_pose[:3],
            )
            
            obs = next_obs
        
        # 计算response指标
        response = self.response_recorder.compute_response()
        
        print(f"  Probe response: force={response['mean_force']:.3f}, "
              f"contact_ratio={response['contact_ratio']:.2f}, "
              f"motion_ratio={response['motion_ratio']:.3f}")
        
        return response
    
    def _generate_probe_action(self, ee_pose: np.ndarray, contact_info: Dict) -> np.ndarray:
        """根据probe配置生成probe动作"""
        if self.probe_config.task_type == "push":
            return self.probe_config.generate_action(
                ee_pose, 
                contact_info['contact_normal']
            )
        elif self.probe_config.task_type == "pick":
            return self.probe_config.generate_action(ee_pose)
        elif self.probe_config.task_type == "pull":
            # Pull方向：远离接触点
            pull_dir = -contact_info['contact_normal']
            return self.probe_config.generate_action(ee_pose, pull_dir)
        else:
            raise ValueError(f"Unknown probe type: {self.probe_config.task_type}")
    
    def _continue_vla_policy(
        self, 
        obs: Dict, 
        trajectory: List, 
        remaining_steps: int
    ) -> str:
        """继续执行VLA策略直到episode结束"""
        
        for step in range(remaining_steps):
            action = self._get_vla_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            trajectory.append({
                'obs': self._extract_obs_dict(obs),
                'action': action,
                'reward': reward,
                'is_contact': False,  # probe后的阶段不再检测contact
            })
            
            obs = next_obs
            
            if terminated or truncated:
                # 判断任务是否成功
                success = info.get('success', False)
                return 'success' if success else 'failure'
        
        # 超时，判断最终状态
        return 'failure'
    
    def _get_vla_action(self, obs: Dict) -> np.ndarray:
        """获取VLA预测的动作"""
        image = self._extract_image(obs)
        instruction = self._get_task_instruction()
        proprio = self._extract_proprio(obs)
        
        action = self.policy.predict(image, instruction, proprio)
        return action
    
    def _get_task_instruction(self) -> str:
        """根据任务名返回自然语言指令"""
        instruction_map = {
            'PushCube-v1': 'push the red cube to the goal',
            'PushChair-v1': 'push the chair forward',
            'PickCube-v1': 'pick up the cube',
            'OpenCabinetDrawer-v1': 'open the drawer',
            'OpenCabinetDoor-v1': 'open the cabinet door',
        }
        return instruction_map.get(self.task_name, f'complete the {self.task_name} task')
    
    def _extract_image(self, obs: Dict) -> np.ndarray:
        """提取RGB图像"""
        # ManiSkill obs['sensor_data']['base_camera']['rgb']
        if 'sensor_data' in obs:
            camera_data = obs['sensor_data']
            # 可能有多个相机，取第一个
            for cam_name in camera_data:
                if 'rgb' in camera_data[cam_name]:
                    return camera_data[cam_name]['rgb']
        
        # 备选：obs['image']
        if 'image' in obs:
            return obs['image']
        
        raise ValueError("Cannot find RGB image in observation")
    
    def _extract_proprio(self, obs: Dict) -> np.ndarray:
        """提取本体感知信息"""
        if 'agent' in obs:
            # ManiSkill的agent state通常包含qpos, qvel等
            qpos = obs['agent'].get('qpos', np.array([]))
            qvel = obs['agent'].get('qvel', np.array([]))
            return np.concatenate([qpos, qvel])
        return np.array([])
    
    def _extract_tcp_force(self, obs: Dict) -> np.ndarray:
        """提取TCP force"""
        if 'extra' in obs and 'tcp_wrench' in obs['extra']:
            return obs['extra']['tcp_wrench'][:3]
        return np.zeros(3)
    
    def _get_ee_pose(self, obs: Dict) -> np.ndarray:
        """获取末端执行器位姿 [x, y, z, qw, qx, qy, qz]"""
        if 'extra' in obs and 'tcp_pose' in obs['extra']:
            return obs['extra']['tcp_pose']
        # 默认返回零向量
        return np.zeros(7)
    
    def _get_object_pose(self, obs: Dict) -> np.ndarray:
        """获取目标物体位姿"""
        # ManiSkill的物体pose通常在obs['extra']['obj_pose']
        if 'extra' in obs and 'obj_pose' in obs['extra']:
            return obs['extra']['obj_pose']
        return np.zeros(7)
    
    def _extract_contacts(self, info: Dict) -> List[Dict]:
        """从info中提取contact信息"""
        if 'contacts' in info:
            return info['contacts']
        return []
    
    def _extract_obs_dict(self, obs: Dict) -> Dict:
        """提取需要保存的观测信息（不保存图像以节省空间）"""
        return {
            'ee_pose': self._get_ee_pose(obs).tolist(),
            'obj_pose': self._get_object_pose(obs).tolist(),
            'tcp_force': self._extract_tcp_force(obs).tolist(),
        }
    
    def _save_episode(self, episode_data: Dict):
        """保存episode数据到HDF5"""
        episode_id = episode_data['episode_id']
        file_path = self.save_dir / f"episode_{episode_id:06d}.hdf5"
        
        with h5py.File(file_path, 'w') as f:
            # 元数据
            f.attrs['episode_id'] = episode_id
            f.attrs['task_name'] = self.task_name
            f.attrs['outcome'] = episode_data['outcome']
            f.attrs['total_steps'] = episode_data['total_steps']
            f.attrs['timestamp'] = episode_data['timestamp']
            
            # Contact image
            if self.save_images and episode_data['contact_image'] is not None:
                f.create_dataset('contact_image', data=episode_data['contact_image'])
            
            # Planned action
            f.create_dataset('planned_action', data=episode_data['planned_action'])
            
            # Probe response
            response_grp = f.create_group('probe_response')
            for key, val in episode_data['probe_response'].items():
                response_grp.create_dataset(key, data=val)
            
            # Trajectory
            traj_grp = f.create_group('trajectory')
            for i, step_data in enumerate(episode_data['trajectory']):
                step_grp = traj_grp.create_group(f'step_{i}')
                step_grp.create_dataset('action', data=step_data['action'])
                step_grp.attrs['reward'] = step_data['reward']
                step_grp.attrs['is_contact'] = step_data['is_contact']
                
                # 保存obs
                for key, val in step_data['obs'].items():
                    step_grp.create_dataset(f'obs/{key}', data=val)
        
        print(f"  Saved to {file_path}")
    
    def get_statistics(self) -> Dict:
        """返回收集统计"""
        return {
            'total_episodes': self.episode_count,
            'successful_episodes': self.success_count,
            'success_rate': self.success_count / max(self.episode_count, 1),
        }
