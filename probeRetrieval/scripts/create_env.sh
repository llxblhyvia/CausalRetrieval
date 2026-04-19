#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/network/rit/lab/wang_lab_cs/yhan/envs"
ENV_DIR="${ENV_ROOT}/probeRetrieval"
PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"

mkdir -p "${ENV_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Load Anaconda/Miniconda first, then rerun this script." >&2
  exit 2
fi

if [[ -d "${ENV_DIR}/conda-meta" ]]; then
  echo "Using existing conda environment: ${ENV_DIR}"
elif [[ -e "${ENV_DIR}" && ! -d "${ENV_DIR}/conda-meta" ]]; then
  echo "Found ${ENV_DIR}, but it is not a conda environment." >&2
  echo "Move or remove that directory first, or set ENV_DIR to another path." >&2
  exit 3
else
  conda create -y -p "${ENV_DIR}" python=3.10 pip
fi

conda run -p "${ENV_DIR}" python -m pip install --upgrade pip
conda run -p "${ENV_DIR}" python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Environment ready: ${ENV_DIR}"
echo "Activate with: conda activate ${ENV_DIR}"
