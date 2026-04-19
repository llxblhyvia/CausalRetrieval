#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval"
cd "${PROJECT_DIR}"
python -m inference.run_inference --config infer_libero_object.yaml "$@"
python -m eval.evaluate --input_dir artifacts
