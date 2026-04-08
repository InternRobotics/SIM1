#!/usr/bin/env bash
# Download SIM1 assets from Hugging Face into ./assets/ (or a custom --local-dir).
# Also downloads dataset references used by DataGen:
#   InternRobotics/Sim1_Dataset -> sim_teleoperated_npz only (via git sparse-checkout).
# Usage:
#   bash download_assets.sh
#   bash download_assets.sh /custom/path

set -e

ASSET_REPO="InternRobotics/Sim1_Assets"
DATASET_REPO="InternRobotics/Sim1_Dataset"
DATASET_SUBDIR="sim_teleoperated_npz"
DEST="${1:-$(dirname "$0")/assets}"

print_sim1_assets_hint() {
    local abs
    if command -v realpath >/dev/null 2>&1; then
        abs="$(realpath "$DEST" 2>/dev/null || echo "$DEST")"
    else
        abs="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"
    fi
    echo "[SIM1] Point SIM1 at this Hugging Face bundle (same as --local-dir above):"
    echo "       export SIM1_ASSETS_ROOT=\"$abs\""
}

download_asset_repo() {
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "[SIM1] Using huggingface-cli for ${ASSET_REPO} ..."
        huggingface-cli download "$ASSET_REPO" \
            --repo-type model \
            --local-dir "$DEST"
        return 0
    fi

    if command -v hf >/dev/null 2>&1; then
        echo "[SIM1] Using hf download for ${ASSET_REPO} ..."
        hf download "$ASSET_REPO" \
            --repo-type model \
            --local-dir "$DEST"
        return 0
    fi

    if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "[SIM1] Using python huggingface_hub for ${ASSET_REPO} ..."
        python3 - "$ASSET_REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:3]
snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
print(f"[SIM1] Downloaded asset repo to: {local_dir}")
PY
        return 0
    fi

    return 1
}

flatten_asset_layout() {
    local nested="$DEST/assets"
    if [[ ! -d "$nested" ]]; then
        return 0
    fi

    echo "[SIM1] Normalizing asset layout into: $DEST"
    local name
    for name in acone cloth random model; do
        if [[ -d "$nested/$name" && ! -e "$DEST/$name" ]]; then
            mv "$nested/$name" "$DEST/$name"
            echo "[SIM1] Moved $nested/$name -> $DEST/$name"
        fi
    done

    # Remove nested dir if empty after move.
    rmdir "$nested" 2>/dev/null || true
}

download_dataset_subset_git() {
    local repo_url="https://huggingface.co/datasets/${DATASET_REPO}"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' RETURN

    echo "[SIM1] Cloning dataset repo (sparse): ${repo_url}"
    git clone --depth 1 --filter=blob:none --sparse "$repo_url" "$tmp_dir/repo"
    git -C "$tmp_dir/repo" sparse-checkout set "$DATASET_SUBDIR"

    # If Git LFS is available, explicitly fetch files under the selected subdir.
    if command -v git-lfs >/dev/null 2>&1 || git -C "$tmp_dir/repo" lfs version >/dev/null 2>&1; then
        git -C "$tmp_dir/repo" lfs pull --include "${DATASET_SUBDIR}/*" || true
    fi

    mkdir -p "$DEST/$DATASET_SUBDIR"
    cp -a "$tmp_dir/repo/$DATASET_SUBDIR/." "$DEST/$DATASET_SUBDIR/"
}

normalize_dataset_npz_layout() {
    # Force dataset layout to:
    #   <DEST>/sim_teleoperated_npz/npz/*.npz
    local root="$DEST/$DATASET_SUBDIR"
    local npz_dir="$root/npz"
    mkdir -p "$npz_dir"

    shopt -s nullglob
    local files=("$root"/*.npz)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        return 0
    fi

    local f
    for f in "${files[@]}"; do
        mv "$f" "$npz_dir/"
    done
    echo "[SIM1] Normalized dataset layout: ${npz_dir}/*.npz"
}

echo "[SIM1] Destination: $DEST"
echo "[SIM1] Asset repo   : $ASSET_REPO"
echo "[SIM1] Dataset repo : $DATASET_REPO (only folder: $DATASET_SUBDIR)"

if ! download_asset_repo; then
    echo "[ERROR] No Hugging Face downloader found (hf, huggingface-cli, or huggingface_hub)."
    echo "  Install with:  pip install -U huggingface_hub"
    echo "  Login with:    hf auth login"
    echo "  Then re-run:   bash download_assets.sh"
    exit 1
fi

flatten_asset_layout

if ! download_dataset_subset_git; then
    echo "[ERROR] Failed to clone dataset subset via git."
    echo "  Repo: https://huggingface.co/datasets/${DATASET_REPO}"
    echo "  Required folder: ${DATASET_SUBDIR}"
    echo "  If repo is private/gated, run: hf auth login"
    exit 1
fi

normalize_dataset_npz_layout

echo "[SIM1] Done. Assets and reference NPZ subset saved to: $DEST"
print_sim1_assets_hint
