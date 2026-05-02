#!/usr/bin/env bash
#SBATCH --job-name=probe-ret-ablate
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/probe-ret-ablate-%A_%a.out
#SBATCH --error=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/probe-ret-ablate-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
cd "${PROJECT_DIR}"

mkdir -p artifacts/slurm_logs

VARIANTS=(
  "full_probe_retrieval"
  "probe_only"
  "random_retrieve"
  "image_only_retrieval"
)

VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_${VARIANT}"

export VARIANT
export WAIT_FOR_WORKERS=1
export EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/goal_probe_retrieval_test_eval_4gpu_video_ablation/${VARIANT}}"
export MEMORY_DIR="${MEMORY_DIR:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank}"
export RESPONSE_JSONL="${RESPONSE_JSONL:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl}"

echo "variant=${VARIANT}"
echo "output_root=${OUTPUT_ROOT}"
echo "memory_dir=${MEMORY_DIR}"
echo "response_jsonl=${RESPONSE_JSONL}"
echo "run_stamp=${RUN_STAMP}"

bash scripts/run_goal_probe_retrieval_test_eval_4gpu.sh \
  --save_video \
  --video_every 1 \
  --video_fps 20
