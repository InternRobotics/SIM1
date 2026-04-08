#!/usr/bin/env bash
# Download SIM1 assets from Hugging Face into ./assets/ (or a custom --local-dir).
# Also downloads dataset references used by DataGen:
#   InternRobotics/Sim1_Dataset -> sim_teleoperated_npz/** only.
# Python and shell scripts resolve the same root via SIM1_ASSETS_ROOT (see sim1_asset_paths.py).
# Usage:
#   bash download_assets.sh                  # default: ./assets
#   bash download_assets.sh /custom/path     # custom destination → then: export SIM1_ASSETS_ROOT=/custom/path

set -e

ASSET_REPO="InternRobotics/Sim1_Assets"
DATASET_REPO="InternRobotics/Sim1_Dataset"
DATASET_PATTERN="sim_teleoperated_npz/**"
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

echo "[SIM1] Destination: $DEST"
echo "[SIM1] Asset repo   : $ASSET_REPO"
echo "[SIM1] Dataset repo : $DATASET_REPO (only: $DATASET_PATTERN)"

# ── Try huggingface-cli (legacy) ──────────────────────────────────────────────
if command -v huggingface-cli &>/dev/null; then
    echo "[SIM1] Using huggingface-cli ..."
    # 1) Core SIM1 assets (full model repo)
    huggingface-cli download "$ASSET_REPO" \
        --repo-type model \
        --local-dir "$DEST"
    # 2) Reference NPZ set for DataGen (dataset subset only)
    huggingface-cli download "$DATASET_REPO" \
        --repo-type dataset \
        --local-dir "$DEST" \
        --include "$DATASET_PATTERN"
    echo "[SIM1] Done. Assets and reference NPZ subset saved to: $DEST"
    print_sim1_assets_hint
    exit 0
fi

# ── Try hf (huggingface_hub ≥ 0.20+; replaces huggingface-cli) ─────────────────
if command -v hf &>/dev/null; then
    echo "[SIM1] Using hf download ..."
    # 1) Core SIM1 assets (full model repo)
    hf download "$ASSET_REPO" \
        --repo-type model \
        --local-dir "$DEST"
    # 2) Reference NPZ set for DataGen (dataset subset only)
    hf download "$DATASET_REPO" \
        --repo-type dataset \
        --local-dir "$DEST" \
        --include "$DATASET_PATTERN"
    echo "[SIM1] Done. Assets and reference NPZ subset saved to: $DEST"
    print_sim1_assets_hint
    exit 0
fi

# ── Fall back to Python huggingface_hub ───────────────────────────────────────
if python3 -c "import huggingface_hub" &>/dev/null; then
    echo "[SIM1] hf / huggingface-cli not found, using Python huggingface_hub ..."
    python3 - "$ASSET_REPO" "$DATASET_REPO" "$DEST" "$DATASET_PATTERN" <<'PY'
import sys
from huggingface_hub import snapshot_download

asset_repo, dataset_repo, local_dir, dataset_pattern = sys.argv[1:5]

snapshot_download(
    repo_id=asset_repo,
    repo_type="model",
    local_dir=local_dir,
)

# Only pull the needed subfolder from dataset repo.
snapshot_download(
    repo_id=dataset_repo,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[dataset_pattern],
)
print(f"[SIM1] Done. Assets and reference NPZ subset saved to: {local_dir}")
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
