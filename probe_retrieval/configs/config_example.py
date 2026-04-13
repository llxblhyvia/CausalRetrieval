"""
配置示例 - 可以根据实际需求调整
"""

# ============================================
# 任务配置
# ============================================

TASKS = {
    # Push类任务
    'PushCube-v1': {
        'episodes': 1000,
        'max_steps': 500,
        'probe_config': {
            'type': 'push',
            'push_distance': 0.015,  # 1.5cm
            'duration_steps': 10,
            'force_threshold': 0.5,
        }
    },
    
    'PushChair-v1': {
        'episodes': 1000,
        'max_steps': 600,
        'probe_config': {
            'type': 'push',
            'push_distance': 0.02,   # 2cm (椅子更重)
            'duration_steps': 12,
            'force_threshold': 0.8,  # 需要更大的力
        }
    },
    
    # Pick类任务
    'PickCube-v1': {
        'episodes': 1000,
        'max_steps': 400,
        'probe_config': {
            'type': 'pick',
            'gripper_close_amount': 0.4,
            'lift_height': 0.003,  # 3mm
            'duration_steps': 10,
            'force_threshold': 0.3,
        }
    },
    
    # Open/Pull类任务
    'OpenCabinetDrawer-v1': {
        'episodes': 1000,
        'max_steps': 500,
        'probe_config': {
            'type': 'pull',
            'pull_distance': 0.02,  # 2cm
            'duration_steps': 12,
            'force_threshold': 0.6,
        }
    },
    
    'OpenCabinetDoor-v1': {
        'episodes': 1000,
        'max_steps': 600,
        'probe_config': {
            'type': 'pull',
            'pull_distance': 0.025,  # 2.5cm
            'duration_steps': 15,
            'force_threshold': 0.7,
        }
    },
}

# ============================================
# GPU配置
# ============================================

GPU_CONFIG = {
    'num_gpus': 4,
    'gpu_ids': [0, 1, 2, 3],
    'envs_per_gpu': 1,  # 每个GPU并行环境数（暂时为1）
}

# ============================================
# VLA配置
# ============================================

VLA_CONFIG = {
    'type': 'openvla',  # 'openvla' 或 'dummy'
    'model_name': 'openvla/openvla-7b',
    'cache_dir': '/path/to/model/cache',  # 可选
}

# ============================================
# 数据保存配置
# ============================================

DATA_CONFIG = {
    'base_dir': './data',
    'save_images': True,  # 是否保存RGB图像
    'compress': False,    # HDF5压缩（暂未实现）
}

# ============================================
# Contact检测配置
# ============================================

CONTACT_CONFIG = {
    'force_threshold': 0.5,      # 默认力阈值
    'min_contact_steps': 2,      # 持续接触步数
    'target_objects': None,      # None=自动检测，或指定列表如['cube', 'chair']
}

# ============================================
# 使用示例
# ============================================

if __name__ == '__main__':
    import json
    
    # 打印配置
    print("Task Configurations:")
    print(json.dumps(TASKS, indent=2))
    
    print("\nGPU Configuration:")
    print(json.dumps(GPU_CONFIG, indent=2))
    
    print("\nVLA Configuration:")
    print(json.dumps(VLA_CONFIG, indent=2))
