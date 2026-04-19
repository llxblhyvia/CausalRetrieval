#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval"
OPENVLA_OFT_REPO="/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft"
LIBERO_REPO="/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO"
HF_HOME="${HF_HOME:-/network/rit/lab/wang_lab_cs/yhan/hf_cache}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/network/rit/lab/wang_lab_cs/yhan/.libero}"

if [[ ! -d "${ENV_DIR}/conda-meta" ]]; then
  echo "Missing conda env: ${ENV_DIR}. Run scripts/create_env.sh first." >&2
  exit 2
fi
if [[ ! -f "${OPENVLA_OFT_REPO}/pyproject.toml" ]]; then
  echo "Missing OpenVLA-OFT repo: ${OPENVLA_OFT_REPO}" >&2
  exit 3
fi
if [[ ! -f "${LIBERO_REPO}/setup.py" ]]; then
  echo "Missing LIBERO repo: ${LIBERO_REPO}" >&2
  exit 4
fi

mkdir -p "${HF_HOME}" "${HF_HOME}/transformers" "${LIBERO_CONFIG_PATH}"
export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
assets: ${LIBERO_REPO}/libero/libero/assets
bddl_files: ${LIBERO_REPO}/libero/libero/bddl_files
benchmark_root: ${LIBERO_REPO}/libero/libero
datasets: ${LIBERO_REPO}/libero/datasets
init_states: ${LIBERO_REPO}/libero/libero/init_files
YAML
fi

conda run -p "${ENV_DIR}" python -m pip install --upgrade pip
conda run -p "${ENV_DIR}" python -m pip install -e "${OPENVLA_OFT_REPO}"
conda run -p "${ENV_DIR}" python -m pip install -e "${LIBERO_REPO}"
conda run -p "${ENV_DIR}" python -m pip install -r "${OPENVLA_OFT_REPO}/experiments/robot/libero/libero_requirements.txt"
conda run -p "${ENV_DIR}" python -m pip install "numpy<2" "opencv-python==4.9.0.80"

echo "Real OpenVLA-OFT/LIBERO dependencies installed in ${ENV_DIR}"
