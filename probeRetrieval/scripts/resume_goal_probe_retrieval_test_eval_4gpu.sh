#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_goal_probe_retrieval_test_eval.sh"
MERGE_SCRIPT="${PROJECT_DIR}/scripts/merge_goal_probe_retrieval_test_eval_shards.sh"

RUN_DIR="${RUN_DIR:-}"
MEMORY_DIR="${MEMORY_DIR:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank}"
RESPONSE_JSONL="${RESPONSE_JSONL:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
VARIANT="${VARIANT:-probe_only}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/existing/run bash $0 [eval args...]" >&2
  echo "   or: bash $0 --run_dir /path/to/existing/run [eval args...]" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

PIDS=()

launch_worker() {
  local gpu="$1"
  local task_names="$2"
  local worker_dir="${RUN_DIR}/gpu${gpu}"
  local log_file="${RUN_DIR}/gpu${gpu}.resume.log"
  shift 2

  mkdir -p "${worker_dir}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TOKENIZERS_PARALLELISM=false \
    bash "${RUN_SCRIPT}" \
      --task_names "${task_names}" \
      --episodes_per_setting "${EPISODES_PER_SETTING}" \
      --memory_dir "${MEMORY_DIR}" \
      --response_jsonl "${RESPONSE_JSONL}" \
      --output_dir "${worker_dir}" \
      --variant "${VARIANT}" \
      --resume \
      "$@" > "${log_file}" 2>&1 < /dev/null &
  local pid="$!"
  PIDS+=("${pid}")
  printf "gpu%s %s\n" "${gpu}" "${pid}" >> "${RUN_DIR}/resume_pids.txt"
}

: > "${RUN_DIR}/resume_pids.txt"
launch_worker 0 push_the_plate_to_the_front_of_the_stove,put_the_cream_cheese_in_the_bowl "$@"
launch_worker 1 put_the_bowl_on_the_stove "$@"
launch_worker 2 put_the_bowl_on_top_of_the_cabinet "$@"
launch_worker 3 put_the_wine_bottle_on_top_of_the_cabinet "$@"

echo "Launched 4 resume workers."
echo "Run directory: ${RUN_DIR}"
echo "PIDs saved to: ${RUN_DIR}/resume_pids.txt"
echo "Logs: ${RUN_DIR}/gpu*.resume.log"

wait "${PIDS[@]}"

echo "Resume workers finished. Merging shards..."
bash "${MERGE_SCRIPT}" --run_dir "${RUN_DIR}"
echo "Merged summary: ${RUN_DIR}/merged/probe_retrieval_test_summary.json"
