#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_probe_goal_physics_sweep.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/probe_goal_physics_sweep_2gpu}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Could not find executable run script at ${RUN_SCRIPT}" >&2
  exit 2
fi

launch_worker() {
  local gpu="$1"
  local task_names="$2"
  local worker_dir="${RUN_DIR}/gpu${gpu}"
  local log_file="${RUN_DIR}/gpu${gpu}.log"

  mkdir -p "${worker_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${RUN_SCRIPT}" \
      --task_names "${task_names}" \
      --episodes_per_setting "${EPISODES_PER_SETTING}" \
      --output_dir "${worker_dir}" \
      "$@" > "${log_file}" 2>&1 &
  echo $!
}

PID0="$(launch_worker 0 push_the_plate_to_the_front_of_the_stove,put_the_cream_cheese_in_the_bowl,put_the_bowl_on_the_stove)"
PID1="$(launch_worker 1 put_the_bowl_on_top_of_the_cabinet,put_the_wine_bottle_on_top_of_the_cabinet)"

cat > "${RUN_DIR}/pids.txt" <<EOF
gpu0 ${PID0}
gpu1 ${PID1}
EOF

cat > "${RUN_DIR}/README.txt" <<EOF
Probe goal physics sweep 2-GPU batch run
run_dir: ${RUN_DIR}
episodes_per_setting: ${EPISODES_PER_SETTING}

Task shards:
  gpu0: push_the_plate_to_the_front_of_the_stove, put_the_cream_cheese_in_the_bowl, put_the_bowl_on_the_stove
  gpu1: put_the_bowl_on_top_of_the_cabinet, put_the_wine_bottle_on_top_of_the_cabinet

Logs:
  ${RUN_DIR}/gpu0.log
  ${RUN_DIR}/gpu1.log

PIDs:
  ${RUN_DIR}/pids.txt
EOF

echo "Started 2 workers."
echo "Run directory: ${RUN_DIR}"
echo "PIDs saved to: ${RUN_DIR}/pids.txt"

wait "${PID0}" "${PID1}"
echo "All workers completed."
