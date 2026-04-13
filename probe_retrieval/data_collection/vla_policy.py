"""
OpenVLA策略封装
"""
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class OpenVLAPolicy:
    """OpenVLA预训练策略"""
    
    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",  # Hugging Face model ID
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: OpenVLA模型名称
            device: 运行设备
            cache_dir: 模型缓存目录
        """
        self.device = device
        self.model_name = model_name
        
        # 延迟加载（避免import时就初始化）
        self.model = None
        self.processor = None
        self.cache_dir = cache_dir
    
    def load(self):
        """加载模型（首次调用时）"""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            print(f"Loading OpenVLA model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            
            self.model.eval()
            print("OpenVLA loaded successfully")
            
        except ImportError:
            raise ImportError(
                "OpenVLA requires transformers. Install with:\n"
                "pip install transformers accelerate"
            )
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,  # RGB image [H, W, 3]
        instruction: str,  # 任务指令
        proprio: Optional[np.ndarray] = None,  # 本体感知 [robot_state_dim]
    ) -> np.ndarray:
        """
        预测动作
        
        Args:
            image: RGB图像
            instruction: 任务描述（如 "pick up the red cube"）
            proprio: 机器人状态（可选）
        
        Returns:
            action: [delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper]
        """
        self.load()  # 确保模型已加载
        
        # 处理输入
        inputs = self.processor(
            text=instruction,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # 如果有proprio信息，添加到输入中（取决于OpenVLA的具体实现）
        if proprio is not None:
            # OpenVLA可能需要将proprio拼接到某个embedding
            # 这里先简化处理，实际需要查看OpenVLA的API
            pass
        
        # 推理
        output = self.model.generate(**inputs, max_new_tokens=50)
        
        # 解码动作
        # OpenVLA输出格式通常是action tokens，需要decode
        action = self._decode_action(output)
        
        return action
    
    def _decode_action(self, output_tokens) -> np.ndarray:
        """
        将模型输出的tokens解码为动作向量
        
        这里需要根据OpenVLA的实际输出格式调整
        """
        # 简化版本：假设OpenVLA直接输出action embeddings
        # 实际可能需要更复杂的解码过程
        
        # Placeholder: 返回一个7维动作
        # 实际应该调用processor的decode方法
        decoded_text = self.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # OpenVLA的action通常编码为文本，如 "[0.1, 0.2, -0.05, 0, 0, 0, 1.0]"
        try:
            # 尝试解析为数值列表
            action_str = decoded_text.strip('[]').split(',')
            action = np.array([float(x) for x in action_str])
            
            # 确保是7维
            if len(action) != 7:
                print(f"Warning: decoded action has {len(action)} dims, expected 7")
                action = np.zeros(7)
            
            return action
        
        except:
            print(f"Warning: failed to decode action from: {decoded_text}")
            return np.zeros(7)  # 失败时返回零动作
    
    def reset(self):
        """重置策略状态（如果有内部状态的话）"""
        pass


class DummyVLAPolicy:
    """
    用于快速测试的dummy策略
    随机生成动作，不需要加载真实模型
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def load(self):
        print("DummyVLAPolicy: No model to load")
    
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回随机动作"""
        # delta_pos: [-0.05, 0.05]
        delta_pos = np.random.uniform(-0.05, 0.05, size=3)
        # delta_rot: [-0.1, 0.1]
        delta_rot = np.random.uniform(-0.1, 0.1, size=3)
        # gripper: [-1, 1]
        gripper = np.random.uniform(-1, 1, size=1)
        
        return np.concatenate([delta_pos, delta_rot, gripper])
    
    def reset(self):
        pass


def create_vla_policy(
    policy_type: str = "openvla",
    device: str = "cuda",
    **kwargs
) -> OpenVLAPolicy:
    """
    Factory函数：创建VLA策略
    
    Args:
        policy_type: "openvla" 或 "dummy"
        device: 运行设备
    """
    if policy_type == "openvla":
        return OpenVLAPolicy(device=device, **kwargs)
    elif policy_type == "dummy":
        return DummyVLAPolicy(device=device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
