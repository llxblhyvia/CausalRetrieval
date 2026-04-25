#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
ENV_DIR="${ENV_DIR:-/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
OPENVLA_OFT_REPO="${OPENVLA_OFT_REPO:-/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft}"
LIBERO_REPO="${LIBERO_REPO:-/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO}"
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b-finetuned-libero-goal}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"
USE_PROPRIO="${USE_PROPRIO:-False}"
USE_L1_REGRESSION="${USE_L1_REGRESSION:-False}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-1}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/logs/goal_baseline}"
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

mkdir -p "${HF_HOME}" "${HF_HOME}/transformers" "${HF_HOME}/hub" "${HF_HOME}/modules" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${LOCAL_LOG_DIR}"
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
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
  --use_proprio "${USE_PROPRIO}" \
  --use_l1_regression "${USE_L1_REGRESSION}" \
  --num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}" \
  --center_crop True \
  --local_log_dir "${LOCAL_LOG_DIR}" \
  "$@"
