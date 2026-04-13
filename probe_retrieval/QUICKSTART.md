# 快速启动指南 (QuickStart)

## 1. 环境安装 (5分钟)

```bash
cd probe_retrieval
bash setup_env.sh
```

按提示选择：
- 是否安装OpenVLA (y/n)
  - 选 `y`: 可以用真实的OpenVLA策略
  - 选 `n`: 只能用dummy策略测试，但安装快

## 2. 测试代码 (1分钟)

```bash
conda activate probe_retrieval
python test_pipeline.py
```

应该看到：
```
✅ All tests passed! Ready to collect data.
```

## 3. 快速收集测试数据 (2分钟)

使用dummy策略快速收集5个episodes测试：

```bash
CUDA_VISIBLE_DEVICES=0 python run_collection.py \
    --task PushCube-v1 \
    --num-episodes 5 \
    --gpus 0 \
    --policy dummy \
    --save-dir ./data/test_5ep
```

输出示例：
```
=== Episode 0 ===
Phase 1: Approaching target...
  ✓ Contact detected at step 23
Phase 2: Executing probe...
  Probe response: force=2.341, contact_ratio=0.80, motion_ratio=0.234
Phase 3: Continuing main policy...
  Outcome: success
  Success rate: 1/1 = 100.00%
  Saved to ./data/test_5ep/gpu_0/episode_000000.hdf5
```

## 4. 检查数据

```bash
# 查看单个episode
python utils/data_inspector.py ./data/test_5ep/gpu_0/episode_000000.hdf5

# 分析整个数据集
python utils/data_inspector.py ./data/test_5ep/
```

## 5. 正式收集 (使用OpenVLA)

### 5.1 单任务，4 GPU并行

```bash
# PushCube: 1000 episodes, ~25分钟
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_collection.py \
    --task PushCube-v1 \
    --num-episodes 1000 \
    --gpus 0,1,2,3 \
    --policy openvla \
    --model-name openvla/openvla-7b \
    --save-dir ./data/pushcube_1k
```

### 5.2 多任务收集

```bash
# 创建批量脚本
cat > collect_all.sh << 'EOF'
#!/bin/bash

TASKS=(
    "PushCube-v1"
    "PushChair-v1"
    "PickCube-v1"
    "OpenCabinetDrawer-v1"
    "OpenCabinetDoor-v1"
)

for TASK in "${TASKS[@]}"; do
    echo "Collecting $TASK..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_collection.py \
        --task $TASK \
        --num-episodes 1000 \
        --gpus 0,1,2,3 \
        --policy openvla \
        --save-dir ./data/${TASK}_1k
done
EOF

chmod +x collect_all.sh
./collect_all.sh
```

## 6. 常见问题

### Q1: CUDA Out of Memory

**解决方案A**: 减少并行环境
```bash
--num-envs-per-gpu 1  # 默认就是1
```

**解决方案B**: 使用更小的模型
```bash
--model-name openvla/openvla-3b  # 替代7b
```

**解决方案C**: 减少GPU数量
```bash
--gpus 0,1  # 只用2张卡
```

### Q2: 没检测到contact

**调试**：在 `data_collection/collector.py` 第132行添加：

```python
print(f"DEBUG: force_mag={force_mag:.3f}, contacts={len(contacts)}")
```

**可能原因**：
1. Force阈值太高 → 降低到0.3
2. 任务本身不需要contact → 换其他任务
3. VLA没接近物体 → 检查VLA是否正常工作

### Q3: VLA加载很慢

**原因**: 首次下载模型 (OpenVLA-7B约14GB)

**解决**：
1. 使用代理或镜像源
2. 提前下载到本地：
```bash
huggingface-cli download openvla/openvla-7b
```

### Q4: 想节省磁盘空间

```bash
# 不保存图像（推荐）
--no-save-images

# 数据量对比：
# 含图像: 1000 episodes × 2MB = 2GB
# 不含图像: 1000 episodes × 50KB = 50MB
```

## 7. 下一步

数据收集完成后，进入 Phase 2：

1. **构建检索库**
   - 提取所有contact_image的特征
   - 提取所有probe_response

2. **Two-Stage Retrieval**
   - Stage 1: Image → topk candidates
   - Stage 2: Probe response → rerank
   - 加权平均得到action_r

3. **Inference测试**
   - Test环境验证
   - 与baseline对比

## 附录：完整参数列表

```bash
python run_collection.py --help

# 主要参数:
--task              # ManiSkill任务名
--num-episodes      # 总episode数
--gpus              # GPU列表，逗号分隔
--policy            # VLA类型: openvla | dummy
--model-name        # OpenVLA模型名
--save-dir          # 数据保存目录
--max-episode-steps # 单episode最大步数
--no-save-images    # 不保存图像
```

## 预期收集时间表

| 任务 | Episodes | GPU数 | 预计时间 |
|------|----------|-------|----------|
| PushCube | 1000 | 4 | 25分钟 |
| PushChair | 1000 | 4 | 30分钟 |
| PickCube | 1000 | 4 | 20分钟 |
| OpenDrawer | 1000 | 4 | 28分钟 |
| **Total** | **5000** | **4** | **~2小时** |

---

**准备好了吗？开始收集数据！** 🚀
