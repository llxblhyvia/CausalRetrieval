#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
RUNNER="${PROJECT_DIR}/scripts/run_goal_probe_retrieval_test_eval_4gpu.sh"
MERGER="${PROJECT_DIR}/scripts/merge_goal_probe_retrieval_test_eval_shards.sh"

EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
VIDEO_FPS="${VIDEO_FPS:-20}"
PROBE_VIDEO_HOLD_FRAMES="${PROBE_VIDEO_HOLD_FRAMES:-10}"
PROBE_VIDEO_FRAME_REPEAT="${PROBE_VIDEO_FRAME_REPEAT:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-artifacts/goal_probe_retrieval_test_eval_4gpu_video_local}"

cd "${PROJECT_DIR}"

run_variant() {
  local variant="$1"
  local output_root="${OUTPUT_BASE}/${variant}_visible_probe"

  mkdir -p "${output_root}"
  echo "============================================================"
  echo "Starting variant: ${variant}"
  echo "Output root: ${output_root}"
  echo "Started at: $(date)"
  echo "============================================================"

  VARIANT="${variant}" \
  OUTPUT_ROOT="${output_root}" \
  EPISODES_PER_SETTING="${EPISODES_PER_SETTING}" \
  WAIT_FOR_WORKERS=1 \
  bash "${RUNNER}" \
    --save_video \
    --video_every 1 \
    --video_fps "${VIDEO_FPS}" \
    --probe_video_hold_frames "${PROBE_VIDEO_HOLD_FRAMES}" \
    --probe_video_frame_repeat "${PROBE_VIDEO_FRAME_REPEAT}"

  local run_dir
  run_dir="$(find "${output_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${run_dir}" ]]; then
    echo "Could not locate latest run directory under ${output_root}" >&2
    exit 3
  fi

  echo "Merging variant ${variant}: ${run_dir}"
  bash "${MERGER}" --run_dir "${run_dir}"
  echo "Finished variant: ${variant}"
  echo "Summary: ${run_dir}/merged/probe_retrieval_test_summary.json"
  echo "Finished at: $(date)"
}

run_variant full_probe_retrieval
run_variant probe_only

echo "All requested visible-probe runs finished at: $(date)"
