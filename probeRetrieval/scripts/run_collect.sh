#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
cd "${PROJECT_DIR}"
python -m rollout.collect_data --config collect_libero_object.yaml "$@"
