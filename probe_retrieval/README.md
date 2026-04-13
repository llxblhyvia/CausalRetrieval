# ProbeRetrieval - Phase 1: Data Collection

多GPU并行收集probe数据，用于两阶段检索系统的训练。

## 核心流程

每个episode的收集流程：

```
1. Reset环境
2. 使用预训练VLA接近目标物体
3. 检测contact (Force + Contact Pair双重验证)
   ├─ 记录contact时刻的image
   └─ 记录VLA计划的action_v
4. 执行probe动作 (task-specific)
   ├─ Push: 轻推1-2cm
   ├─ Pick: 轻夹+微提2mm
   └─ Pull: 轻拉1-2cm
   └─ 记录response (8个指标)
5. 继续VLA策略至episode结束
6. 记录outcome (success/failure)
```

## 安装

```bash
# 创建环境
conda create -n probe_retrieval python=3.10
conda activate probe_retrieval

# 安装依赖
pip install -r requirements.txt

# 安装ManiSkill (GPU版本)
pip install mani-skill[all]
```

## 快速开始

### 1. 测试运行 (Dummy VLA)

使用dummy策略快速测试整个pipeline：

```bash
cd probe_retrieval

# 单GPU测试，收集10个episodes
CUDA_VISIBLE_DEVICES=0 python run_collection.py \
    --task PushCube-v1 \
    --num-episodes 10 \
    --gpus 0 \
    --policy dummy \
    --save-dir ./data/test_run
```

### 2. 正式收集 (4 GPU并行)

使用OpenVLA策略收集大规模数据：

```bash
# 4张A100并行，每张收集250个episodes (总共1000)
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_collection.py \
    --task PushCube-v1 \
    --num-episodes 1000 \
    --gpus 0,1,2,3 \
    --policy openvla \
    --model-name openvla/openvla-7b \
    --save-dir ./data/pushcube_1k \
    --max-episode-steps 500
```

### 3. 其他任务

```bash
# PickCube
python run_collection.py --task PickCube-v1 --num-episodes 1000 ...

# OpenDrawer
python run_collection.py --task OpenCabinetDrawer-v1 --num-episodes 1000 ...

# PushChair
python run_collection.py --task PushChair-v1 --num-episodes 1000 ...
```

## 数据格式

每个episode保存为一个HDF5文件：

```
episode_000001.hdf5
├── attributes
│   ├── episode_id: 1
│   ├── task_name: "PushCube-v1"
│   ├── outcome: "success" or "failure"
│   └── total_steps: 245
├── contact_image [H, W, 3]  # RGB图像
├── planned_action [7]        # VLA计划的动作
├── probe_response/
│   ├── mean_force
│   ├── max_force
│   ├── force_std
│   ├── contact_ratio
│   ├── total_ee_motion
│   ├── total_obj_motion
│   ├── motion_ratio
│   └── probe_steps
└── trajectory/
    ├── step_0/
    │   ├── action [7]
    │   ├── obs/ee_pose [7]
    │   ├── obs/obj_pose [7]
    │   └── ...
    └── ...
```

## 数据检查

```bash
# 检查单个episode
python utils/data_inspector.py data/pushcube_1k/gpu_0/episode_000001.hdf5

# 分析整个数据集
python utils/data_inspector.py data/pushcube_1k/

# 输出示例:
# Total Episodes: 1000
#   Success: 750
#   Failure: 250
#   Success Rate: 75.00%
# 
# Probe Response Metrics:
#   mean_force:
#     Mean: 2.3451 ± 0.8234
#     Range: [0.5123, 5.6789]
#   contact_ratio:
#     Mean: 0.8234 ± 0.1123
#   ...
```

## 配置文件

### Probe动作配置

在 `configs/probe_actions.py` 中定义不同任务的probe策略：

```python
TASK_TO_PROBE_CONFIG = {
    "PushCube-v1": PushProbeConfig(
        push_distance=0.015,  # 1.5cm
        duration_steps=10,
    ),
    "PickCube-v1": PickProbeConfig(
        gripper_close_amount=0.4,
        lift_height=0.003,  # 3mm
        duration_steps=10,
    ),
    ...
}
```

### Contact检测阈值

在收集器初始化时调整：

```python
collector = ProbeDataCollector(
    ...,
    force_threshold=0.5,  # 力阈值 (N)
    min_contact_steps=2,  # 持续2步才算有效接触
)
```

## 多GPU并行策略

每张GPU独立运行，数据保存到各自目录：

```
data/
├── gpu_0/
│   ├── episode_000000.hdf5
│   ├── episode_000001.hdf5
│   └── ...
├── gpu_1/
│   └── ...
├── gpu_2/
│   └── ...
├── gpu_3/
│   └── ...
└── collection_stats.json
```

优势：
- 线性加速（4 GPU = 4x速度）
- 无GPU间通信开销
- 单卡故障不影响其他卡

## 估算收集时间

假设条件：
- 单个episode平均200步
- VLA推理: ~20ms/step
- Probe: 10步
- 环境step: ~10ms/step

单episode时间：
```
200 steps × (20ms VLA + 10ms env) + 10 steps × 10ms probe
= 200 × 30ms + 100ms
= 6.1 seconds
```

收集1000 episodes：
- 单GPU: 1000 × 6.1s ≈ 1.7小时
- 4 GPU并行: 1.7 / 4 ≈ **25分钟**

## 预期数据规模

1000 episodes × 每个约2MB (含图像) = **2GB**

如果不保存图像 (`--no-save-images`)：
1000 episodes × 约50KB = **50MB**

## Troubleshooting

### CUDA Out of Memory

```bash
# 减少并行环境数
--num-envs-per-gpu 1

# 或者使用更小的VLA模型
--model-name openvla/openvla-3b
```

### Contact检测不到

检查：
1. force阈值是否太高
2. 任务中是否真的有接触（有些任务可能不需要接触）
3. ManiSkill环境的contact返回格式

调试：
```python
# 在collector.py中添加打印
print(f"TCP force: {tcp_force}, magnitude: {force_mag}")
print(f"Contacts: {contacts}")
```

### VLA策略加载失败

确保：
1. Hugging Face token已配置 (如果模型是私有的)
2. 网络连接正常
3. 磁盘空间充足 (OpenVLA-7B约14GB)

## 下一步

Phase 2: Two-Stage Retrieval Inference
- Image-based初步检索 → topk candidates
- Probe-based re-ranking
- 加权平均获得action_r

## Citation

如果使用了OpenVLA:
```bibtex
@article{openvla2024,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```
