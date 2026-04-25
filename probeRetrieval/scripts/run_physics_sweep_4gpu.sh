#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
ENV_DIR="${ENV_DIR:-/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/physics_sweep_4gpu}"

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m diagnostics.physics_sweep --task_ids 0,1,2 --output_dir "${OUTPUT_DIR}/gpu0" "$@" &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m diagnostics.physics_sweep --task_ids 3,4,5 --output_dir "${OUTPUT_DIR}/gpu1" "$@" &
PID1=$!
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -m diagnostics.physics_sweep --task_ids 6,7 --output_dir "${OUTPUT_DIR}/gpu2" "$@" &
PID2=$!
CUDA_VISIBLE_DEVICES=3 "${PYTHON}" -m diagnostics.physics_sweep --task_ids 8,9 --output_dir "${OUTPUT_DIR}/gpu3" "$@" &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
