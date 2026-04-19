#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
ENV_DIR="${ENV_DIR:-/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
OPENVLA_OFT_REPO="${OPENVLA_OFT_REPO:-/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft}"
LIBERO_REPO="${LIBERO_REPO:-/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO}"
CHECKPOINT="${CHECKPOINT:-/network/rit/lab/wang_lab_cs/yhan/repos/openvla-7b-oft}"
HF_HOME="/network/rit/lab/wang_lab_cs/yhan/hf_cache"
XDG_CACHE_HOME="/network/rit/lab/wang_lab_cs/yhan/cache"
MPLCONFIGDIR="/network/rit/lab/wang_lab_cs/yhan/cache/matplotlib"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Could not find Python at ${PYTHON}" >&2
  echo "Run: bash scripts/create_env.sh" >&2
  exit 2
fi
if [[ ! -f "${OPENVLA_OFT_REPO}/experiments/robot/libero/run_libero_eval.py" ]]; then
  echo "Could not find OpenVLA-OFT eval script under ${OPENVLA_OFT_REPO}" >&2
  exit 2
fi
if [[ ! -d "${LIBERO_REPO}/libero/libero" ]]; then
  echo "Could not find LIBERO package under ${LIBERO_REPO}/libero/libero" >&2
  exit 2
fi

mkdir -p "${HF_HOME}" "${HF_HOME}/transformers" "${HF_HOME}/hub" "${HF_HOME}/modules" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
export HF_HOME
export XDG_CACHE_HOME
export MPLCONFIGDIR
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/network/rit/lab/wang_lab_cs/yhan/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
assets: ${LIBERO_REPO}/libero/libero/assets
bddl_files: ${LIBERO_REPO}/libero/libero/bddl_files
benchmark_root: ${LIBERO_REPO}/libero/libero
datasets: ${LIBERO_REPO}/libero/datasets
init_states: ${LIBERO_REPO}/libero/libero/init_files
YAML
fi
export PYTHONPATH="${LIBERO_REPO}:${OPENVLA_OFT_REPO}:${PYTHONPATH:-}"
cd "${OPENVLA_OFT_REPO}"
"${PYTHON}" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_object \
  --center_crop True \
  "$@"
