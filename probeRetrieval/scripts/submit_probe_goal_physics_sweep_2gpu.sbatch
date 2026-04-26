#!/usr/bin/env bash
#SBATCH --job-name=probe-goal-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/%x-%j.out
#SBATCH --error=/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
ENV_DIR="${ENV_DIR:-/network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval}"
RUN_SCRIPT="${PROJECT_DIR}/scripts/run_probe_goal_physics_sweep_2gpu_batch.sh"
SLURM_LOG_DIR="${PROJECT_DIR}/artifacts/slurm_logs"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/probe_goal_physics_sweep_2gpu}"
EPISODES_PER_SETTING="${EPISODES_PER_SETTING:-10}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-nojob}}"

mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}"

cd "${PROJECT_DIR}"

if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_DIR}/bin/activate"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TIMESTAMP=${TIMESTAMP}"
echo "EPISODES_PER_SETTING=${EPISODES_PER_SETTING}"

ARGS=(
  --frictions "${FRICTIONS:-0.2,0.7}"
  --mass_scales "${MASS_SCALES:-0.5,3.0,7.0}"
  --max_steps "${MAX_STEPS:-300}"
)

if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
  ARGS+=(--save_video)
fi
if [[ -n "${VIDEO_EVERY:-}" ]]; then
  ARGS+=(--video_every "${VIDEO_EVERY}")
fi
if [[ -n "${VIDEO_FPS:-}" ]]; then
  ARGS+=(--video_fps "${VIDEO_FPS}")
fi

OUTPUT_ROOT="${OUTPUT_ROOT}" \
EPISODES_PER_SETTING="${EPISODES_PER_SETTING}" \
TIMESTAMP="${TIMESTAMP}" \
bash "${RUN_SCRIPT}" "${ARGS[@]}"
