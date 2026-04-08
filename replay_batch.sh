#!/bin/bash
# ============================================================
# SIM1-Sim Batch Replay
#
# Replays all .npz trajectory files in a given directory in one run,
# writing paired outputs under replay/<FOLDER_NAME>_NNNN/{npz,usd}/.
#
# Usage:
#   bash replay_batch.sh [DATASET_DIR] [FOLDER_NAME]
#
# Arguments:
#   DATASET_DIR  - Directory containing .npz files (default: ./dataset/example/gen/kf)
#   FOLDER_NAME  - Session base name under replay/ (default: replay_output)
#                  Creates e.g. replay/replay_output_0001/npz/ and .../usd/
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_DIR="${1:-${PROJECT_ROOT}/dataset/example/gen/kf}"
FOLDER_NAME="${2:-replay_output}"

echo "============================================================"
echo "SIM1-Sim Batch Replay"
echo "============================================================"
echo "Dataset dir:  $DATASET_DIR"
echo "Session base: $FOLDER_NAME  ->  replay/${FOLDER_NAME}_NNNN/"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python "${PROJECT_ROOT}/apps/replay_app.py" "$DATASET_DIR" --folder_name "$FOLDER_NAME"

echo "Batch replay complete."
