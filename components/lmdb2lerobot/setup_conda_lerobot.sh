#!/usr/bin/env bash
# Create conda environment 'lerobot' (Python 3.12) and install all dependencies in one shot.
#
#   bash components/lmdb2lerobot/setup_conda_lerobot.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lerobot"

# ── locate conda ──────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "Error: conda not found. Please install Miniconda or Anaconda first." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# ── create environment (skip if already exists) ───────────────────────────────
if [[ ! -x "${PIP}" ]]; then
  echo ">>> Creating conda environment '${ENV_NAME}' (python=3.12) ..."
  conda create -y -n "${ENV_NAME}" python=3.12 pip
else
  echo ">>> Environment '${ENV_NAME}' already exists, skipping creation."
fi

# ── install LeRobot (includes torch) ─────────────────────────────────────────
echo ">>> Installing Hugging Face lerobot (includes torch; may take 10-30 min on first run) ..."
"${PIP}" install "git+https://github.com/huggingface/lerobot.git"

# ── install extra conversion dependencies ────────────────────────────────────
echo ">>> Installing conversion dependencies (opencv / lmdb / imageio-ffmpeg ...) ..."
"${PIP}" install -r "${SCRIPT_DIR}/requirements.txt"

# ── sanity check ─────────────────────────────────────────────────────────────
echo ">>> Sanity check ..."
"${PY}" -c "
import cv2, lmdb, torch, imageio
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    layout = 'lerobot.common (legacy)'
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    layout = 'lerobot.datasets (modern)'
print('  cv2    :', cv2.__version__)
print('  torch  :', torch.__version__)
print('  lerobot:', layout, '-> LeRobotDataset OK')
print('All dependencies ready.')
"

echo ""
echo "Setup complete. Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
