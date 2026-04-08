#!/usr/bin/env bash
# Download SIM1 assets from Hugging Face into ./assets/
# Usage:
#   bash download_assets.sh                  # default: ./assets
#   bash download_assets.sh /custom/path     # custom destination

set -e

REPO="InternRobotics/Sim1_Assets"
DEST="${1:-$(dirname "$0")/assets}"

echo "[SIM1] Downloading assets from HuggingFace: $REPO"
echo "[SIM1] Destination: $DEST"

# ── Try huggingface-cli first (fastest, supports resume) ──────────────────────
if command -v huggingface-cli &>/dev/null; then
    echo "[SIM1] Using huggingface-cli ..."
    huggingface-cli download "$REPO" \
        --repo-type model \
        --local-dir "$DEST"
    echo "[SIM1] Done. Assets saved to: $DEST"
    exit 0
fi

# ── Fall back to Python huggingface_hub ───────────────────────────────────────
if python3 -c "import huggingface_hub" &>/dev/null; then
    echo "[SIM1] huggingface-cli not found, using Python huggingface_hub ..."
    python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
)
print(f"[SIM1] Done. Assets saved to: {local_dir}")
PY
    exit 0
fi

# ── Neither available: prompt user to install ─────────────────────────────────
echo "[ERROR] Neither huggingface-cli nor huggingface_hub Python package found."
echo "  Install with:  pip install huggingface_hub"
echo "  Then re-run:   bash download_assets.sh"
exit 1
