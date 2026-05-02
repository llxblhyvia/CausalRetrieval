#!/usr/bin/env bash
#SBATCH --job-name=probe-ret-video
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/probe-ret-video-%j.out
#SBATCH --error=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/probe-ret-video-%j.err

set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
cd "${PROJECT_DIR}"

mkdir -p artifacts/slurm_logs

export VARIANT="${VARIANT:-full_probe_retrieval}"
export WAIT_FOR_WORKERS=1
export EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/goal_probe_retrieval_test_eval_4gpu_video_full}"
export MEMORY_DIR="${MEMORY_DIR:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank}"
export RESPONSE_JSONL="${RESPONSE_JSONL:-artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl}"

bash scripts/run_goal_probe_retrieval_test_eval_4gpu.sh \
  --save_video \
  --video_every 1 \
  --video_fps 20
