#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_goal_probe_memory_collect.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/collect_goal_probe_memory_sweep_4gpu}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Could not find run script at ${RUN_SCRIPT}" >&2
  exit 2
fi

launch_worker() {
  local gpu="$1"
  local task_names="$2"
  local worker_dir="${RUN_DIR}/gpu${gpu}"
  local log_file="${RUN_DIR}/gpu${gpu}.log"
  shift 2

  mkdir -p "${worker_dir}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${RUN_SCRIPT}" \
      --task_names "${task_names}" \
      --episodes_per_setting "${EPISODES_PER_SETTING}" \
      --output_dir "${worker_dir}" \
      --memory_dir "${worker_dir}/memory_bank" \
      "$@" > "${log_file}" 2>&1 < /dev/null &
  echo $!
}

PID0="$(launch_worker 0 push_the_plate_to_the_front_of_the_stove,put_the_cream_cheese_in_the_bowl)"
PID1="$(launch_worker 1 put_the_bowl_on_the_stove)"
PID2="$(launch_worker 2 put_the_bowl_on_top_of_the_cabinet)"
PID3="$(launch_worker 3 put_the_wine_bottle_on_top_of_the_cabinet)"

cat > "${RUN_DIR}/pids.txt" <<EOF
gpu0 ${PID0}
gpu1 ${PID1}
gpu2 ${PID2}
gpu3 ${PID3}
EOF

cat > "${RUN_DIR}/README.txt" <<EOF
Goal probe memory collection 4-GPU run
run_dir: ${RUN_DIR}
episodes_per_setting: ${EPISODES_PER_SETTING}
frictions: 0.2, 0.7
mass_scales: 0.5, 3.0, 7.0

Task shards:
  gpu0: push_the_plate_to_the_front_of_the_stove, put_the_cream_cheese_in_the_bowl
  gpu1: put_the_bowl_on_the_stove
  gpu2: put_the_bowl_on_top_of_the_cabinet
  gpu3: put_the_wine_bottle_on_top_of_the_cabinet

Each shard writes:
  episodes.jsonl
  summary.json
  collection_summary.json
  memory_bank/metadata.jsonl
  memory_bank/arrays.npz
EOF

echo "Launched 4 workers in background."
echo "Run directory: ${RUN_DIR}"
echo "PIDs saved to: ${RUN_DIR}/pids.txt"
