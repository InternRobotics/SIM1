#!/bin/bash
# ============================================================
# SIM1 dependency installation (setup.sh)
#
# Installs all dependencies for SIM1-DataGen and SIM1-Sim,
# including MuJoCo, Warp, and the Newton physics engine.
#
# Prerequisites:
#   - A conda environment activated with Python 3.11
#     conda create -n sim1 python=3.11 -y && conda activate sim1
#   - CUDA toolkit >= 11.8
#
# Reference:
#   https://newton-physics.github.io/newton/0.2.2/guide/installation.html#method-3-manual-setup-using-pip-in-a-virtual-environment
#
# Usage:
#   conda activate sim1
#   bash setup.sh
#
# PyTorch wheels (optional override):
#   Default: CUDA 12.x (cu124 index). For CPU-only:
#     TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup.sh
#
# Render pipeline (components/render/main.py Steps 1–4, MeisterRender submodule) is installed
# by default into the same conda env: yourdfpy, imageio, lmdb, bpy, omegaconf, minexr, opencv-python, …
#   See components/render/README.md for package → step mapping.
# To skip (e.g. bpy wheel unavailable on your platform):
#   SIM1_SKIP_RENDER=1 bash setup.sh
#
# What gets installed (exact pip lines are in the script body; README does not duplicate this list):
#   - mujoco, mujoco-warp, warp-lang (NVIDIA pre-release index)
#   - Newton from ./newton (pip install -e ".[dev]")
#   - torch, torchvision, torchaudio (TORCH_INDEX_URL, default CUDA 12.4 / cu124)
#   - einops, rotary-embedding-torch, diffusers (DataGen --use_dp)
#   - numpy, scipy, matplotlib, opencv-python-headless, Pillow, tqdm, lmdb, imageio[ffmpeg], huggingface_hub
#   - unless SIM1_SKIP_RENDER=1: components/render/requirement.txt + MeisterRender/requirements.txt
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  SIM1 Installation"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ─── Step 1: Install MuJoCo and Warp ───────────────────────
echo "[Step 1/5] Installing MuJoCo..."
python -m pip install mujoco

echo ""
echo "[Step 2/5] Installing MuJoCo-Warp..."
python -m pip install mujoco-warp

echo ""
echo "[Step 3/5] Installing NVIDIA Warp (pre-release)..."
python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/

# ─── Step 2: Install Newton (editable mode) ────────────────
echo ""
echo "[Step 4/5] Installing Newton physics engine (editable)..."

NEWTON_DIR="${PROJECT_ROOT}/newton"

if [ ! -d "${NEWTON_DIR}" ] || [ ! -f "${NEWTON_DIR}/pyproject.toml" ]; then
    echo "Error: Newton source not found at ${NEWTON_DIR}"
    echo "Please ensure the newton/ directory contains the Newton source code."
    echo "  Expected: ${NEWTON_DIR}/pyproject.toml"
    exit 1
fi

cd "${NEWTON_DIR}"
python -m pip install -e ".[dev]"
cd "${PROJECT_ROOT}"

# ─── Step 3: PyTorch (DataGen / diffusion pipeline) ────────
echo ""
echo "[Step 5/5] Installing PyTorch (torch, torchvision, torchaudio)..."
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "${TORCH_INDEX_URL}"

echo ""
echo "[Info] Installing DataGen --use_dp dependencies (einops, rotary-embedding-torch, diffusers)..."
python -m pip install einops rotary-embedding-torch diffusers

# ─── Step 4: Other Python dependencies ─────────────────────
echo ""
echo "[Info] Installing additional dependencies..."
# lmdb + imageio: required by components/render/step4_render_acone.py (always import before Step 4)
# huggingface_hub: required by download_assets.sh (asset download from HuggingFace)
python -m pip install numpy scipy matplotlib opencv-python-headless Pillow tqdm lmdb "imageio[ffmpeg]" huggingface_hub

# ─── Render stack (default): main.py Steps 1–4 + MeisterRender / step4_render_acone.py ──
if [ "${SIM1_SKIP_RENDER:-0}" = "1" ]; then
    echo ""
    echo "[Info] Skipping render pipeline install (SIM1_SKIP_RENDER=1)."
else
    echo ""
    echo "============================================================"
    echo "  SIM1 render stack (components/render/) — default install"
    echo "  pip: requirement.txt (+ MeisterRender/requirements.txt)"
    echo "============================================================"
    RENDER_REQ="${PROJECT_ROOT}/components/render/requirement.txt"
    if [ ! -f "${RENDER_REQ}" ]; then
        echo "  Warning: Missing ${RENDER_REQ}"
    else
        echo ""
        echo "[Render] pip install -r components/render/requirement.txt ..."
        python -m pip install -r "${RENDER_REQ}"
        echo ""
        echo "[Render] Verifying MeisterRender (step4_render_acone.py) import ..."
        python -c "
import os, sys
root = r'${PROJECT_ROOT}/components/render/MeisterRender'
sys.path.insert(0, root)
from api import RenderEngine
print('  MeisterRender RenderEngine OK')
" || echo "  Warning: MeisterRender import check failed."
    fi
fi

# ─── Verify installation ───────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying Installation"
echo "============================================================"

echo ""
echo "[Verify] Checking Newton import..."
python -c "import newton; print('  Newton version:', newton.__version__)" 2>/dev/null || {
    echo "  Warning: Could not import newton. Please check the installation."
}

echo ""
echo "[Verify] Checking Warp import..."
python -c "import warp as wp; print('  Warp OK')" 2>/dev/null || {
    echo "  Warning: Could not import warp."
}

echo ""
echo "[Verify] Checking PyTorch..."
python -c "import torch; import torchvision; print('  torch:', torch.__version__, '| cuda:', torch.cuda.is_available())" 2>/dev/null || {
    echo "  Warning: Could not import torch/torchvision."
}

echo ""
echo "[Verify] Checking DataGen DP stack (einops, rotary_embedding_torch, diffusers)..."
python -c "import einops; import rotary_embedding_torch; import diffusers; print('  einops / rotary_embedding_torch / diffusers OK')" 2>/dev/null || {
    echo "  Warning: Could not import einops, rotary_embedding_torch, or diffusers."
}

echo ""
echo "[Verify] Checking huggingface_hub (required by download_assets.sh)..."
python -c "import huggingface_hub; print('  huggingface_hub', huggingface_hub.__version__, 'OK')" 2>/dev/null || {
    echo "  Warning: Could not import huggingface_hub."
}

echo ""
echo "[Verify] Checking Python version..."
python --version

if [ "${SIM1_SKIP_RENDER:-0}" != "1" ]; then
    echo ""
    echo "[Verify] Render pipeline (yourdfpy, imageio, lmdb, Step 4 / MeisterRender) ..."
    python -c "import yourdfpy; import imageio; import lmdb; print('  yourdfpy / imageio / lmdb OK')" 2>/dev/null || {
        echo "  Warning: yourdfpy, imageio, or lmdb import failed."
    }
fi

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "Quick start (run from project root: cd \"${PROJECT_ROOT}\"):"
echo ""
echo "  # Interactive teleoperation (requires display/OpenGL)"
echo "  python apps/teleoperation_app.py --task lift_manip_shirt"
echo ""
echo "  # Data generation"
echo "  python apps/datagen_app.py --data_folder ./dataset/example --num 2 --use_dp"
echo ""
echo "  # Replay: headless (default, no window)"
echo "  python apps/replay_app.py ./dataset/example/gen/000000.npz"
echo ""
echo "  # Replay: OpenGL window (visualize)"
echo "  python apps/replay_app.py ./dataset/example/gen/000000.npz --no-headless"
echo ""
echo "  # Replay: window + save MP4 (--save_video requires --no-headless)"
echo "  python apps/replay_app.py ./dataset/example/gen/000000.npz --no-headless --save_video"
echo ""
echo "  # Replay outputs: replay/<folder_name>_NNNN/{npz,usd,video}/ (NNNN auto-increments)"
echo ""
echo "  # Render: Steps 1–3 (default), then batch Step 4"
echo "  python components/render/main.py --root_dir ./replay/your_session_0001"
echo "  bash components/render/batch_step4.sh ./replay/your_session_0001"
echo ""
echo "  # Skip render packages during setup: SIM1_SKIP_RENDER=1 bash setup.sh"
echo ""
echo "For more details, see README.md and components/render/README.md"
