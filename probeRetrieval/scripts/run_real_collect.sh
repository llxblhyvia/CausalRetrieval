#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
ENV_DIR="${ENV_DIR:-/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"

cd "${PROJECT_DIR}"
"${PYTHON}" -m rollout.collect_data \
  --config collect_libero_object.yaml \
  --output_dir artifacts/real_collect \
  --memory_dir artifacts/real_collect/memory_bank \
  "$@"
