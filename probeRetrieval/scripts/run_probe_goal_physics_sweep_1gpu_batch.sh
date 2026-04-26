#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_probe_goal_physics_sweep.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/probe_goal_physics_sweep_1gpu}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Could not find executable run script at ${RUN_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/gpu0"

echo "Started 1 worker."
echo "Run directory: ${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0 \
bash "${RUN_SCRIPT}" \
  --task_names "push_the_plate_to_the_front_of_the_stove,put_the_cream_cheese_in_the_bowl,put_the_bowl_on_the_stove,put_the_bowl_on_top_of_the_cabinet,put_the_wine_bottle_on_top_of_the_cabinet" \
  --episodes_per_setting "${EPISODES_PER_SETTING}" \
  --output_dir "${RUN_DIR}/gpu0" \
  "$@" 2>&1 | tee "${RUN_DIR}/gpu0.log"

echo "Worker completed."
