#!/usr/bin/env bash
# Download SIM1 assets from Hugging Face into ./assets/ (or a custom --local-dir).
# Python and shell scripts resolve the same root via SIM1_ASSETS_ROOT (see sim1_asset_paths.py).
# Usage:
#   bash download_assets.sh                  # default: ./assets
#   bash download_assets.sh /custom/path     # custom destination → then: export SIM1_ASSETS_ROOT=/custom/path

set -e

REPO="InternRobotics/Sim1_Assets"
DEST="${1:-$(dirname "$0")/assets}"

print_sim1_assets_hint() {
    local abs
    if command -v realpath &>/dev/null; then
        abs="$(realpath "$DEST" 2>/dev/null || echo "$DEST")"
    else
        abs="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"
    fi
    echo "[SIM1] Point SIM1 at this Hugging Face bundle (same as --local-dir above):"
    echo "       export SIM1_ASSETS_ROOT=\"$abs\""
}

echo "[SIM1] Downloading assets from HuggingFace: $REPO"
echo "[SIM1] Destination: $DEST"

# ── Try huggingface-cli (legacy) ──────────────────────────────────────────────
if command -v huggingface-cli &>/dev/null; then
    echo "[SIM1] Using huggingface-cli ..."
    huggingface-cli download "$REPO" \
        --repo-type model \
        --local-dir "$DEST"
    echo "[SIM1] Done. Assets saved to: $DEST"
    print_sim1_assets_hint
    exit 0
fi

# ── Try hf (huggingface_hub ≥ 0.20+; replaces huggingface-cli) ─────────────────
if command -v hf &>/dev/null; then
    echo "[SIM1] Using hf download ..."
    hf download "$REPO" \
        --repo-type model \
        --local-dir "$DEST"
    echo "[SIM1] Done. Assets saved to: $DEST"
    print_sim1_assets_hint
    exit 0
fi

# ── Fall back to Python huggingface_hub ───────────────────────────────────────
if python3 -c "import huggingface_hub" &>/dev/null; then
    echo "[SIM1] hf / huggingface-cli not found, using Python huggingface_hub ..."
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
    print_sim1_assets_hint
    exit 0
fi

# ── Neither available: prompt user to install ─────────────────────────────────
echo "[ERROR] No Hugging Face downloader found (hf, huggingface-cli, or huggingface_hub)."
echo "  Install with:  pip install -U huggingface_hub"
echo "  Login with:    hf auth login"
echo "  Then re-run:   bash download_assets.sh"
exit 1
