#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_goal_baseline.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/goal_baseline_4gpu}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Could not find executable run script at ${RUN_SCRIPT}" >&2
  exit 2
fi

launch_worker() {
  local gpu="$1"
  local task_ids="$2"
  local worker_dir="${RUN_DIR}/gpu${gpu}"
  local log_file="${RUN_DIR}/gpu${gpu}.log"

  mkdir -p "${worker_dir}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    NUM_TRIALS_PER_TASK="${EPISODES_PER_TASK}" \
    LOCAL_LOG_DIR="${worker_dir}/logs" \
    bash "${RUN_SCRIPT}" --task_ids "${task_ids}" > "${log_file}" 2>&1 < /dev/null &
  echo $!
}

PID0="$(launch_worker 0 0,1,2)"
PID1="$(launch_worker 1 3,4,5)"
PID2="$(launch_worker 2 6,7)"
PID3="$(launch_worker 3 8,9)"

cat > "${RUN_DIR}/pids.txt" <<EOF
gpu0 ${PID0}
gpu1 ${PID1}
gpu2 ${PID2}
gpu3 ${PID3}
EOF

cat > "${RUN_DIR}/README.txt" <<EOF
Goal baseline 4-GPU run
run_dir: ${RUN_DIR}
episodes_per_task: ${EPISODES_PER_TASK}

Logs:
  ${RUN_DIR}/gpu0.log
  ${RUN_DIR}/gpu1.log
  ${RUN_DIR}/gpu2.log
  ${RUN_DIR}/gpu3.log

PIDs:
  ${RUN_DIR}/pids.txt
EOF

echo "Launched 4 workers in background."
echo "Run directory: ${RUN_DIR}"
echo "PIDs saved to: ${RUN_DIR}/pids.txt"
echo "Tail logs with:"
echo "  tail -f ${RUN_DIR}/gpu0.log ${RUN_DIR}/gpu1.log ${RUN_DIR}/gpu2.log ${RUN_DIR}/gpu3.log"
