# ProbeRetrieval 项目结构

```
probe_retrieval/
├── README.md                          # 完整项目文档
├── QUICKSTART.md                      # 快速启动指南（重点看这个！）
├── requirements.txt                   # Python依赖
├── setup_env.sh                       # 环境安装脚本
├── run_collection.py                  # 【主入口】多GPU数据收集
├── test_pipeline.py                   # 测试脚本
│
├── configs/                           # 配置文件
│   ├── probe_actions.py              # ✨ Probe动作定义（三类任务）
│   └── config_example.py             # 配置示例
│
├── data_collection/                   # 数据收集模块
│   ├── collector.py                  # ✨ 核心收集器
│   └── vla_policy.py                 # VLA策略加载（OpenVLA + Dummy）
│
└── utils/                             # 工具模块
    ├── contact_detection.py          # ✨ Contact检测 + Response记录
    └── data_inspector.py             # 数据验证和可视化
```

## 核心文件说明

### ✨ 三个最重要的文件

1. **configs/probe_actions.py**
   - 定义三类任务的probe动作
   - Push: 轻推1-2cm
   - Pick: 轻夹+微提2mm
   - Pull: 轻拉1-2cm

2. **utils/contact_detection.py**
   - Contact双重验证（Force + Contact Pair）
   - 记录8个response指标
   - 完整的probe数据记录

3. **data_collection/collector.py**
   - 完整的episode收集流程
   - VLA接近 → Contact检测 → Probe执行 → 继续VLA
   - HDF5数据保存

### 主要脚本

- **run_collection.py**: 多GPU并行收集入口
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_collection.py \
      --task PushCube-v1 \
      --num-episodes 1000 \
      --gpus 0,1,2,3 \
      --policy openvla
  ```

- **test_pipeline.py**: 测试所有模块
  ```bash
  python test_pipeline.py
  ```

- **utils/data_inspector.py**: 数据分析工具
  ```bash
  python utils/data_inspector.py ./data/pushcube_1k/
  ```

## 数据流程

```
Episode收集流程:
┌──────────────┐
│ Reset环境    │
└──────┬───────┘
       ▼
┌──────────────────────────┐
│ Phase 1: VLA接近目标     │
│  - 每步检测contact       │
│  - Force + Contact Pair  │
└──────┬───────────────────┘
       ▼ Contact detected!
┌──────────────────────────┐
│ 记录contact image        │
│ 记录planned action_v     │
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Phase 2: 执行Probe       │
│  - Push/Pick/Pull动作    │
│  - 记录response (8指标)  │
│  - duration_steps步      │
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Phase 3: 继续VLA至结束   │
│  - 跑完整个episode       │
│  - 记录outcome           │
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ 保存HDF5文件             │
│  - contact_image         │
│  - planned_action        │
│  - probe_response        │
│  - trajectory            │
│  - outcome               │
└──────────────────────────┘
```

## HDF5数据格式

```python
episode_000001.hdf5
{
    # 元数据
    'attrs': {
        'episode_id': 1,
        'task_name': 'PushCube-v1',
        'outcome': 'success',  # or 'failure'
        'total_steps': 245,
    },
    
    # Contact时刻的图像
    'contact_image': np.ndarray[H, W, 3],
    
    # VLA计划的动作
    'planned_action': np.ndarray[7],
    
    # Probe response (8个指标)
    'probe_response': {
        'mean_force': 2.5,
        'max_force': 3.2,
        'force_std': 0.8,
        'contact_ratio': 0.8,
        'total_ee_motion': 0.015,
        'total_obj_motion': 0.0075,
        'motion_ratio': 0.5,
        'probe_steps': 10,
    },
    
    # 完整轨迹
    'trajectory': {
        'step_0': {
            'action': [7],
            'obs/ee_pose': [7],
            'obs/obj_pose': [7],
            'reward': 0.0,
            'attrs': {'is_contact': False},
        },
        'step_1': {...},
        ...
    }
}
```

## 关键设计决策

### 1. Contact检测：双重验证
```python
valid_contact = (force > threshold) AND (contact_pair_exists) AND (持续N步)
```

### 2. Probe动作：Task-specific
- **Push**: 沿接触法向量反方向推
- **Pick**: 轻微闭合gripper + 小幅上提
- **Pull**: 沿拉方向移动 + gripper抓紧

### 3. Response指标：8个维度
1. Force统计: mean, max, std
2. Contact比例: contact_ratio
3. 运动量: ee_motion, obj_motion, motion_ratio
4. Probe步数: probe_steps

### 4. 多GPU并行：独立进程
- 每GPU独立运行
- 无GPU间通信
- 数据保存到各自目录
- 最后merge统计

## 使用流程

### 快速测试（5分钟）
```bash
# 1. 安装环境
bash setup_env.sh

# 2. 测试代码
python test_pipeline.py

# 3. 收集测试数据（5个episodes）
CUDA_VISIBLE_DEVICES=0 python run_collection.py \
    --task PushCube-v1 \
    --num-episodes 5 \
    --gpus 0 \
    --policy dummy \
    --save-dir ./data/test_5ep

# 4. 检查数据
python utils/data_inspector.py ./data/test_5ep/
```

### 正式收集（2小时收集5个任务各1000 episodes）
```bash
# 使用OpenVLA，4 GPU并行
./collect_all.sh  # 见QUICKSTART.md
```

## 下一步：Phase 2 Implementation

完成数据收集后，你需要实现：

1. **特征提取**
   - Contact image → visual embedding (CLIP/ResNet)
   - Probe response → 8维向量

2. **检索系统**
   - Stage 1: Image相似度 → topk=20
   - Stage 2: Response相似度 → rerank → topk=5

3. **动作融合**
   ```python
   action_r = weighted_avg(topk_actions)
   final_action = alpha * action_r + (1-alpha) * action_v
   ```

4. **Inference Pipeline**
   - 集成到实际测试环境
   - 对比baseline (纯VLA)

## 估算

- **时间**: 5任务 × 1000ep × 6s/ep ÷ 4 GPU ≈ 2小时
- **数据量**: 5000ep × 2MB = **10GB** (含图像)
- **成功率**: 预期60-80%（取决于VLA质量）

---

**准备开始了吗？查看 QUICKSTART.md！** 🚀
