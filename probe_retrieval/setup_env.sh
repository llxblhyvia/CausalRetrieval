#!/bin/bash
# 自动安装ProbeRetrieval环境

set -e  # 遇到错误立即退出

echo "========================================="
echo "ProbeRetrieval Environment Setup"
echo "========================================="
echo ""

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✓ Conda found"

# 创建环境
ENV_NAME="probe_retrieval"
ENV_PREFIX="/network/rit/lab/wang_lab_cs/yhan/envs/probe_retrieval"
PYTHON_VERSION="3.10"

echo ""
echo "Creating conda environment at: $ENV_PREFIX (Python $PYTHON_VERSION)"
conda create -p $ENV_PREFIX python=$PYTHON_VERSION -y

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_PREFIX

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "✓ CUDA detected: $CUDA_VERSION"
else
    echo "⚠ CUDA not detected. Will install CPU-only PyTorch."
fi

# 安装PyTorch
echo ""
echo "Installing PyTorch..."
if [[ $CUDA_VERSION == 12.* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ $CUDA_VERSION == 11.* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision
fi

# 安装基础依赖
echo ""
echo "Installing base dependencies..."
pip install numpy>=1.24.0 gymnasium>=0.29.0 h5py>=3.8.0 matplotlib>=3.7.0

# 安装ManiSkill
echo ""
echo "Installing ManiSkill..."
pip install mani-skill>=3.0.0

# 安装OpenVLA依赖（可选）
read -p "Install OpenVLA dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing transformers and accelerate..."
    pip install transformers>=4.36.0 accelerate>=0.25.0 pillow>=10.0.0
    echo "✓ OpenVLA dependencies installed"
else
    echo "⊘ Skipping OpenVLA (you can use --policy dummy for testing)"
fi

# 验证安装
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python -c "import gymnasium; print(f'✓ Gymnasium {gymnasium.__version__}')"
python -c "import h5py; print(f'✓ h5py {h5py.__version__}')"
python -c "import mani_skill; print(f'✓ ManiSkill {mani_skill.__version__}')" || echo "⚠ ManiSkill not found"

# 检查transformers（可选）
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "⊘ Transformers not installed (OpenVLA unavailable)"

echo ""
echo "========================================="
echo "✅ Environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_PREFIX"
echo ""
echo "To test the installation:"
echo "  python test_pipeline.py"
echo ""
echo "To start collecting data:"
echo "  CUDA_VISIBLE_DEVICES=0 python run_collection.py --task PushCube-v1 --num-episodes 10 --policy dummy"
echo ""