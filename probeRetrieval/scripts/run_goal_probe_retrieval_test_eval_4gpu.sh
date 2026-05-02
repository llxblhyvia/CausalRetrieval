#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_goal_probe_retrieval_test_eval.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/goal_probe_retrieval_test_eval_4gpu}"
MEMORY_DIR="${MEMORY_DIR:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank}"
RESPONSE_JSONL="${RESPONSE_JSONL:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
VARIANT="${VARIANT:-full_probe_retrieval}"
WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-0}"
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
      --memory_dir "${MEMORY_DIR}" \
      --response_jsonl "${RESPONSE_JSONL}" \
      --output_dir "${worker_dir}" \
      --variant "${VARIANT}" \
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
Goal probe retrieval test eval 4-GPU run
run_dir: ${RUN_DIR}
variant: ${VARIANT}
episodes_per_setting: ${EPISODES_PER_SETTING}
memory_dir: ${MEMORY_DIR}
response_jsonl: ${RESPONSE_JSONL}
test_frictions: 0.05, 0.5
test_mass_scales: 0.05, 5.0, 10.0

Task shards:
  gpu0: push_the_plate_to_the_front_of_the_stove, put_the_cream_cheese_in_the_bowl
  gpu1: put_the_bowl_on_the_stove
  gpu2: put_the_bowl_on_top_of_the_cabinet
  gpu3: put_the_wine_bottle_on_top_of_the_cabinet

Each shard writes:
  probe_retrieval_test_episodes.jsonl
  probe_retrieval_test_summary.json
EOF

echo "Launched 4 workers in background."
echo "Run directory: ${RUN_DIR}"
echo "PIDs saved to: ${RUN_DIR}/pids.txt"
if [[ "${WAIT_FOR_WORKERS}" == "1" ]]; then
  echo "Waiting for workers to finish..."
  wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
  echo "All workers finished."
fi
