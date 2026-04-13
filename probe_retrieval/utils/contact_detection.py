"""
Contact检测：Force sensor + Contact pair双重验证
"""
import numpy as np
from typing import Optional, List, Dict, Tuple


class ContactDetector:
    """检测机器人末端与目标物体的有效接触"""
    
    def __init__(
        self,
        force_threshold: float = 0.5,  # 力阈值 (N)
        min_contact_steps: int = 2,  # 最小持续接触步数
        target_object_names: Optional[List[str]] = None,
    ):
        """
        Args:
            force_threshold: TCP力大小阈值
            min_contact_steps: 持续接触多少步才算有效
            target_object_names: 目标物体名称列表（如 ["cube", "chair"]）
        """
        self.force_threshold = force_threshold
        self.min_contact_steps = min_contact_steps
        self.target_object_names = target_object_names or []
        
        # 内部状态
        self.contact_step_counter = 0
        self.contact_history = []
    
    def detect_contact(
        self, 
        obs: Dict,
        env_contacts: List[Dict],  # ManiSkill返回的contact信息
    ) -> Tuple[bool, Optional[Dict]]:
        """
        检测是否发生有效接触
        
        Args:
            obs: 环境观测，包含force信息
            env_contacts: ManiSkill的contact pairs列表
                每个contact: {
                    'body1': str,  # 例如 "panda_hand"
                    'body2': str,  # 例如 "cube"
                    'force': np.ndarray,  # 接触力向量
                }
        
        Returns:
            (is_contact, contact_info)
            contact_info = {
                'force_magnitude': float,
                'force_vector': np.ndarray,
                'contact_normal': np.ndarray,
                'contact_point': np.ndarray,
                'target_object': str,
            }
        """
        # 1. 检查force magnitude
        tcp_force = self._extract_tcp_force(obs)
        force_mag = np.linalg.norm(tcp_force)
        
        force_satisfied = force_mag > self.force_threshold
        
        # 2. 检查contact pair
        ee_contact_info = self._check_ee_object_contact(env_contacts)
        pair_satisfied = ee_contact_info is not None
        
        # 3. 双重条件都满足
        if force_satisfied and pair_satisfied:
            self.contact_step_counter += 1
            contact_info = {
                'force_magnitude': force_mag,
                'force_vector': tcp_force,
                'contact_normal': ee_contact_info['normal'],
                'contact_point': ee_contact_info['point'],
                'target_object': ee_contact_info['object_name'],
            }
            self.contact_history.append(contact_info)
        else:
            self.contact_step_counter = 0
        
        # 4. 判断是否达到持续接触阈值
        is_valid_contact = self.contact_step_counter >= self.min_contact_steps
        
        if is_valid_contact and self.contact_history:
            return True, self.contact_history[-1]
        else:
            return False, None
    
    def _extract_tcp_force(self, obs: Dict) -> np.ndarray:
        """从观测中提取TCP force"""
        # ManiSkill obs结构: obs['extra']['tcp_wrench'] = [fx, fy, fz, tx, ty, tz]
        if 'extra' in obs and 'tcp_wrench' in obs['extra']:
            wrench = obs['extra']['tcp_wrench']
            return wrench[:3]  # 只要force部分
        
        # 备选：从agent state中获取
        if 'agent' in obs and 'tcp_force' in obs['agent']:
            return obs['agent']['tcp_force']
        
        # 如果都没有，返回零向量
        return np.zeros(3)
    
    def _check_ee_object_contact(self, contacts: List[Dict]) -> Optional[Dict]:
        """
        检查末端执行器与目标物体的接触
        
        Returns:
            contact_info = {
                'normal': np.ndarray,  # 接触法向量
                'point': np.ndarray,   # 接触点
                'object_name': str,
            }
        """
        ee_keywords = ['hand', 'gripper', 'finger', 'tcp', 'end_effector']
        
        for contact in contacts:
            body1 = contact.get('body1', '').lower()
            body2 = contact.get('body2', '').lower()
            
            # 检查是否一个是ee，另一个是目标物体
            ee_in_1 = any(kw in body1 for kw in ee_keywords)
            ee_in_2 = any(kw in body2 for kw in ee_keywords)
            
            target_in_1 = self._is_target_object(body1)
            target_in_2 = self._is_target_object(body2)
            
            if (ee_in_1 and target_in_2) or (ee_in_2 and target_in_1):
                # 找到了ee和目标物体的接触
                force_vec = contact.get('force', np.zeros(3))
                normal = force_vec / (np.linalg.norm(force_vec) + 1e-8)
                
                return {
                    'normal': normal,
                    'point': contact.get('position', np.zeros(3)),
                    'object_name': body2 if target_in_2 else body1,
                }
        
        return None
    
    def _is_target_object(self, body_name: str) -> bool:
        """判断是否是目标物体"""
        if not self.target_object_names:
            # 如果没有指定，则任何非ee物体都算
            ee_keywords = ['hand', 'gripper', 'finger', 'tcp', 'robot', 'panda', 'base']
            return not any(kw in body_name.lower() for kw in ee_keywords)
        
        return any(target in body_name.lower() for target in self.target_object_names)
    
    def reset(self):
        """重置检测器状态"""
        self.contact_step_counter = 0
        self.contact_history.clear()


class ProbeResponseRecorder:
    """记录probe阶段的response数据"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置记录器"""
        self.forces = []  # 每步的force magnitude
        self.contact_flags = []  # 每步是否接触
        self.ee_positions = []  # 末端位置序列
        self.obj_positions = []  # 物体位置序列
    
    def record_step(
        self,
        force_magnitude: float,
        is_contact: bool,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
    ):
        """记录单步数据"""
        self.forces.append(force_magnitude)
        self.contact_flags.append(is_contact)
        self.ee_positions.append(ee_pos.copy())
        self.obj_positions.append(obj_pos.copy())
    
    def compute_response(self) -> Dict[str, float]:
        """
        计算8个response指标
        
        Returns:
            {
                'mean_force': float,
                'max_force': float,
                'force_std': float,
                'contact_ratio': float,  # 有接触的步数比例
                'total_ee_motion': float,  # 末端总位移
                'total_obj_motion': float,  # 物体总位移
                'motion_ratio': float,  # obj_motion / ee_motion
                'probe_steps': int,  # probe总步数
            }
        """
        if len(self.forces) == 0:
            return self._empty_response()
        
        forces_arr = np.array(self.forces)
        contacts_arr = np.array(self.contact_flags, dtype=float)
        
        # Force统计
        mean_force = forces_arr.mean()
        max_force = forces_arr.max()
        force_std = forces_arr.std()
        
        # Contact比例
        contact_ratio = contacts_arr.mean()
        
        # 末端运动量
        ee_positions_arr = np.array(self.ee_positions)
        ee_displacements = np.diff(ee_positions_arr, axis=0)
        total_ee_motion = np.linalg.norm(ee_displacements, axis=1).sum()
        
        # 物体运动量
        obj_positions_arr = np.array(self.obj_positions)
        obj_displacements = np.diff(obj_positions_arr, axis=0)
        total_obj_motion = np.linalg.norm(obj_displacements, axis=1).sum()
        
        # 运动比例
        motion_ratio = total_obj_motion / (total_ee_motion + 1e-8)
        
        return {
            'mean_force': float(mean_force),
            'max_force': float(max_force),
            'force_std': float(force_std),
            'contact_ratio': float(contact_ratio),
            'total_ee_motion': float(total_ee_motion),
            'total_obj_motion': float(total_obj_motion),
            'motion_ratio': float(motion_ratio),
            'probe_steps': len(self.forces),
        }
    
    def _empty_response(self) -> Dict[str, float]:
        """返回空的response（用于异常情况）"""
        return {
            'mean_force': 0.0,
            'max_force': 0.0,
            'force_std': 0.0,
            'contact_ratio': 0.0,
            'total_ee_motion': 0.0,
            'total_obj_motion': 0.0,
            'motion_ratio': 0.0,
            'probe_steps': 0,
        }
