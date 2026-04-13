"""
Probe动作定义：针对不同任务类型设计特定的probe策略
"""
import numpy as np
from typing import Dict, Tuple


class ProbeActionConfig:
    """Probe动作的基类配置"""
    
    def __init__(
        self,
        duration_steps: int = 10,  # probe持续多少步
        force_threshold: float = 0.5,  # 接触力阈值 (N)
        contact_duration: int = 2,  # 持续接触步数才算有效
    ):
        self.duration_steps = duration_steps
        self.force_threshold = force_threshold
        self.contact_duration = contact_duration


class PushProbeConfig(ProbeActionConfig):
    """推动类任务的probe配置"""
    
    def __init__(self, push_distance: float = 0.015, **kwargs):
        """
        Args:
            push_distance: 轻推距离，单位米 (默认1.5cm)
        """
        super().__init__(**kwargs)
        self.push_distance = push_distance
        self.task_type = "push"
    
    def generate_action(self, current_ee_pose: np.ndarray, contact_normal: np.ndarray) -> np.ndarray:
        """
        生成轻推动作
        
        Args:
            current_ee_pose: 当前末端位置 [x, y, z, qw, qx, qy, qz]
            contact_normal: 接触法向量（从物体指向末端）
        
        Returns:
            action: [delta_x, delta_y, delta_z, delta_rot, gripper]
        """
        # 沿着接触法向量的反方向轻推（进入物体方向）
        push_direction = -contact_normal / (np.linalg.norm(contact_normal) + 1e-8)
        delta_pos = push_direction * self.push_distance
        
        # 保持当前姿态，gripper维持开启
        delta_rot = np.zeros(3)  # 不旋转
        gripper = 1.0  # 开启状态
        
        return np.concatenate([delta_pos, delta_rot, [gripper]])


class PickProbeConfig(ProbeActionConfig):
    """抓取类任务的probe配置"""
    
    def __init__(
        self, 
        gripper_close_amount: float = 0.3,  # gripper闭合量 [0-1]
        lift_height: float = 0.002,  # 轻微上提高度 (2mm)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gripper_close_amount = gripper_close_amount
        self.lift_height = lift_height
        self.task_type = "pick"
    
    def generate_action(self, current_ee_pose: np.ndarray, **kwargs) -> np.ndarray:
        """
        生成轻夹+微提动作
        
        Returns:
            action: [delta_x, delta_y, delta_z, delta_rot, gripper]
        """
        # 轻微上提
        delta_pos = np.array([0.0, 0.0, self.lift_height])
        delta_rot = np.zeros(3)
        
        # gripper部分闭合
        gripper = 1.0 - self.gripper_close_amount
        
        return np.concatenate([delta_pos, delta_rot, [gripper]])


class PullProbeConfig(ProbeActionConfig):
    """拉/开门类任务的probe配置"""
    
    def __init__(self, pull_distance: float = 0.015, **kwargs):
        """
        Args:
            pull_distance: 轻拉距离 (默认1.5cm)
        """
        super().__init__(**kwargs)
        self.pull_distance = pull_distance
        self.task_type = "pull"
    
    def generate_action(self, current_ee_pose: np.ndarray, pull_direction: np.ndarray) -> np.ndarray:
        """
        生成轻拉动作
        
        Args:
            current_ee_pose: 当前末端位置
            pull_direction: 拉的方向向量（通常是远离接触点）
        
        Returns:
            action: [delta_x, delta_y, delta_z, delta_rot, gripper]
        """
        # 沿着拉方向移动
        pull_vec = pull_direction / (np.linalg.norm(pull_direction) + 1e-8)
        delta_pos = pull_vec * self.pull_distance
        
        delta_rot = np.zeros(3)
        gripper = -1.0  # 抓紧状态（拉的时候需要抓住）
        
        return np.concatenate([delta_pos, delta_rot, [gripper]])


# 任务名称到probe配置的映射
TASK_TO_PROBE_CONFIG = {
    # Push类
    "PushCube-v1": PushProbeConfig(push_distance=0.015, duration_steps=10),
    "PushChair-v1": PushProbeConfig(push_distance=0.02, duration_steps=12),
    "PushT-v1": PushProbeConfig(push_distance=0.015, duration_steps=10),
    
    # Pick类
    "PickCube-v1": PickProbeConfig(gripper_close_amount=0.4, lift_height=0.003),
    "PickSingleYCB-v1": PickProbeConfig(gripper_close_amount=0.35, lift_height=0.002),
    "PegInsertionSide-v1": PickProbeConfig(gripper_close_amount=0.5, lift_height=0.001),
    
    # Pull/Open类
    "OpenCabinetDrawer-v1": PullProbeConfig(pull_distance=0.02, duration_steps=12),
    "OpenCabinetDoor-v1": PullProbeConfig(pull_distance=0.025, duration_steps=15),
    "TurnFaucet-v1": PullProbeConfig(pull_distance=0.01, duration_steps=8),
}


def get_probe_config(task_name: str) -> ProbeActionConfig:
    """根据任务名获取对应的probe配置"""
    if task_name in TASK_TO_PROBE_CONFIG:
        return TASK_TO_PROBE_CONFIG[task_name]
    
    # 默认策略：根据任务名推断
    task_lower = task_name.lower()
    if any(keyword in task_lower for keyword in ["push", "move"]):
        return PushProbeConfig()
    elif any(keyword in task_lower for keyword in ["pick", "grasp", "place"]):
        return PickProbeConfig()
    elif any(keyword in task_lower for keyword in ["open", "pull", "turn"]):
        return PullProbeConfig()
    else:
        # 默认使用push probe
        print(f"Warning: Unknown task {task_name}, using default PushProbe")
        return PushProbeConfig()
