#!/usr/bin/env bash
# Download SIM1 assets from Hugging Face into ./assets/ (or a custom --local-dir).
# Also downloads dataset references used by DataGen:
#   InternRobotics/Sim1_Dataset -> sim_teleoperated_npz only.
# Usage:
#   bash download_assets.sh
#   bash download_assets.sh /custom/path

set -e

ASSET_REPO="InternRobotics/Sim1_Assets"
DATASET_REPO="InternRobotics/Sim1_Dataset"
DATASET_SUBDIR="sim_teleoperated_npz"
DEST="${1:-$(dirname "$0")/assets}"

has_valid_npz_dataset() {
    local npz_dir="$DEST/$DATASET_SUBDIR/npz"
    if [[ ! -d "$npz_dir" ]]; then
        return 1
    fi
    shopt -s nullglob
    local npz_files=("$npz_dir"/*.npz)
    shopt -u nullglob
    if [[ ${#npz_files[@]} -eq 0 ]]; then
        return 1
    fi
    # Reject Git LFS pointer placeholders.
    local sample="${npz_files[0]}"
    local header
    header="$(dd if="$sample" bs=1 count=64 2>/dev/null | tr -d '\000' || true)"
    [[ "$header" != version\ https://git-lfs.github.com/spec/v1* ]]
}

build_missing_asset_patterns() {
    # Print one pattern per line.
    # We request both flat and nested forms to handle repo layout differences.
    if [[ ! -d "$DEST/acone" ]]; then
        echo "acone/*"
        echo "assets/acone/*"
    fi
    if [[ ! -d "$DEST/cloth" ]]; then
        echo "cloth/*"
        echo "assets/cloth/*"
    fi
    if [[ ! -d "$DEST/random" ]]; then
        echo "random/*"
        echo "assets/random/*"
    fi
    if [[ ! -f "$DEST/model/flow_ckpt_three.pth" ]]; then
        echo "model/*"
        echo "assets/model/*"
    fi
}

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
    local stage="$DEST/.hf_assets_download"
    local -a include_patterns=()
    while IFS= read -r p; do
        [[ -n "$p" ]] && include_patterns+=("$p")
    done < <(build_missing_asset_patterns)

    if [[ ${#include_patterns[@]} -eq 0 ]]; then
        echo "[SIM1] Core assets already present under $DEST; skipping asset repo download."
        return 0
    fi

    echo "[SIM1] Missing asset groups detected; downloading only required parts..."
    rm -rf "$stage"
    mkdir -p "$stage"

    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "[SIM1] Using huggingface-cli for ${ASSET_REPO} ..."
        huggingface-cli download "$ASSET_REPO" --repo-type model --local-dir "$stage" \
            $(for p in "${include_patterns[@]}"; do printf -- "--include %q " "$p"; done)
        merge_downloaded_assets "$stage"
        return 0
    fi

    if command -v hf >/dev/null 2>&1; then
        echo "[SIM1] Using hf download for ${ASSET_REPO} ..."
        hf download "$ASSET_REPO" --repo-type model --local-dir "$stage" \
            $(for p in "${include_patterns[@]}"; do printf -- "--include %q " "$p"; done)
        merge_downloaded_assets "$stage"
        return 0
    fi

    if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "[SIM1] Using python huggingface_hub for ${ASSET_REPO} ..."
        python3 - "$ASSET_REPO" "$stage" "${include_patterns[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, *patterns = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=patterns if patterns else None,
)
print(f"[SIM1] Downloaded asset repo to: {local_dir}")
PY
        merge_downloaded_assets "$stage"
        return 0
    fi

    return 1
}

merge_downloaded_assets() {
    local stage="$1"
    local src="$stage"
    if [[ -d "$stage/assets" ]]; then
        # HF repo may contain a top-level "assets/" folder.
        src="$stage/assets"
    fi

    mkdir -p "$DEST"
    shopt -s dotglob nullglob
    local entries=("$src"/*)
    shopt -u dotglob nullglob

    local item base target
    for item in "${entries[@]}"; do
        base="$(basename "$item")"
        target="$DEST/$base"
        if [[ -e "$target" && -d "$item" && -d "$target" ]]; then
            # Merge directories but keep existing files to avoid repeated full-copy conflicts.
            cp -a -n "$item/." "$target/" || true
            rmdir "$item" 2>/dev/null || true
        elif [[ -e "$target" ]]; then
            rm -rf "$item"
        else
            mv "$item" "$target"
        fi
    done

    rm -rf "$stage"
}

ensure_model_folder() {
    local model_dir="$DEST/model"
    if [[ -d "$model_dir" ]]; then
        return 0
    fi

    echo "[SIM1] Missing model folder after asset sync; trying targeted download (model/*) ..."
    local stage="$DEST/.hf_model_download"
    rm -rf "$stage"
    mkdir -p "$stage"

    local ok=0
    if command -v huggingface-cli >/dev/null 2>&1; then
        if huggingface-cli download "$ASSET_REPO" --repo-type model --include "model/*" --local-dir "$stage"; then
            ok=1
        elif huggingface-cli download "$ASSET_REPO" --repo-type model --include "assets/model/*" --local-dir "$stage"; then
            ok=1
        fi
    elif command -v hf >/dev/null 2>&1; then
        if hf download "$ASSET_REPO" --repo-type model --include "model/*" --local-dir "$stage"; then
            ok=1
        elif hf download "$ASSET_REPO" --repo-type model --include "assets/model/*" --local-dir "$stage"; then
            ok=1
        fi
    elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        if python3 - "$ASSET_REPO" "$stage" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1:3]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    allow_patterns=["model/*", "assets/model/*"],
)
PY
        then
            ok=1
        fi
    fi

    if [[ "$ok" -eq 1 ]]; then
        merge_downloaded_assets "$stage"
    else
        rm -rf "$stage"
    fi

    [[ -d "$model_dir" ]]
}

flatten_asset_layout() {
    local nested="$DEST/assets"
    if [[ ! -d "$nested" ]]; then
        return 0
    fi

    echo "[SIM1] Normalizing asset layout into: $DEST"
    # Merge all children under <DEST>/assets/ into <DEST>/ (handles both known and
    # future folders/files, and avoids leaving a nested assets/assets layout).
    shopt -s dotglob nullglob
    local entries=("$nested"/*)
    shopt -u dotglob nullglob

    if [[ ${#entries[@]} -eq 0 ]]; then
        rmdir "$nested" 2>/dev/null || true
        return 0
    fi

    local src base target
    for src in "${entries[@]}"; do
        base="$(basename "$src")"
        target="$DEST/$base"

        if [[ -e "$target" ]]; then
            if [[ -d "$src" && -d "$target" ]]; then
                # Merge directory contents to avoid creating DEST/assets/*.
                shopt -s dotglob nullglob
                local sub_entries=("$src"/*)
                shopt -u dotglob nullglob
                if [[ ${#sub_entries[@]} -gt 0 ]]; then
                    local sub
                    for sub in "${sub_entries[@]}"; do
                        mv "$sub" "$target/"
                    done
                fi
                rmdir "$src" 2>/dev/null || true
            else
                # Keep existing target; drop duplicate downloaded copy.
                rm -rf "$src"
            fi
        else
            mv "$src" "$target"
        fi
    done

    # Remove nested dir if empty after merge.
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

    # Dataset files are stored with Git LFS; without LFS we only get pointer text files.
    if command -v git-lfs >/dev/null 2>&1 || git -C "$tmp_dir/repo" lfs version >/dev/null 2>&1; then
        git -C "$tmp_dir/repo" lfs pull --include "${DATASET_SUBDIR}/*"
    else
        echo "[ERROR] git-lfs is required for dataset sparse clone path but is not available."
        return 1
    fi

    mkdir -p "$DEST/$DATASET_SUBDIR"
    cp -a "$tmp_dir/repo/$DATASET_SUBDIR/." "$DEST/$DATASET_SUBDIR/"
}

download_dataset_subset_hf() {
    if has_valid_npz_dataset; then
        echo "[SIM1] Dataset subset already present under $DEST/$DATASET_SUBDIR/npz; skipping dataset download."
        return 0
    fi

    local stage="$DEST/.hf_dataset_download"
    rm -rf "$stage"
    mkdir -p "$stage"

    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "[SIM1] Using huggingface-cli for dataset subset ${DATASET_REPO}/${DATASET_SUBDIR} ..."
        huggingface-cli download "$DATASET_REPO" \
            --repo-type dataset \
            --include "${DATASET_SUBDIR}/*" \
            --local-dir "$stage"
    elif command -v hf >/dev/null 2>&1; then
        echo "[SIM1] Using hf download for dataset subset ${DATASET_REPO}/${DATASET_SUBDIR} ..."
        hf download "$DATASET_REPO" \
            --repo-type dataset \
            --include "${DATASET_SUBDIR}/*" \
            --local-dir "$stage"
    elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "[SIM1] Using python huggingface_hub for dataset subset ${DATASET_REPO}/${DATASET_SUBDIR} ..."
        python3 - "$DATASET_REPO" "$DATASET_SUBDIR" "$stage" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, subset, local_dir = sys.argv[1:4]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[f"{subset}/*"],
)
print(f"[SIM1] Downloaded dataset subset to: {local_dir}")
PY
    else
        rm -rf "$stage"
        return 1
    fi

    mkdir -p "$DEST/$DATASET_SUBDIR"
    if [[ -d "$stage/$DATASET_SUBDIR" ]]; then
        cp -a "$stage/$DATASET_SUBDIR/." "$DEST/$DATASET_SUBDIR/"
    else
        echo "[ERROR] Dataset subset not found after HF download: $stage/$DATASET_SUBDIR"
        rm -rf "$stage"
        return 1
    fi
    rm -rf "$stage"
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

verify_final_layout() {
    # Required by SIM1 runtime/data-gen:
    #   <DEST>/acone/acone.urdf
    #   <DEST>/cloth/short-shirt.usdc
    #   <DEST>/model/flow_ckpt_three.pth
    #   <DEST>/sim_teleoperated_npz/npz/*.npz
    local missing=0

    if [[ ! -f "$DEST/acone/acone.urdf" ]]; then
        echo "[ERROR] Missing required file: $DEST/acone/acone.urdf"
        missing=1
    fi
    if [[ ! -f "$DEST/cloth/short-shirt.usdc" ]]; then
        echo "[ERROR] Missing required file: $DEST/cloth/short-shirt.usdc"
        missing=1
    fi
    if [[ ! -f "$DEST/model/flow_ckpt_three.pth" ]]; then
        echo "[ERROR] Missing required file: $DEST/model/flow_ckpt_three.pth"
        missing=1
    fi

    local npz_dir="$DEST/$DATASET_SUBDIR/npz"
    if [[ ! -d "$npz_dir" ]]; then
        echo "[ERROR] Missing required directory: $npz_dir"
        missing=1
    else
        shopt -s nullglob
        local npz_files=("$npz_dir"/*.npz)
        shopt -u nullglob
        if [[ ${#npz_files[@]} -eq 0 ]]; then
            echo "[ERROR] No reference NPZ found under: $npz_dir"
            missing=1
        else
            # Guard against Git LFS pointer text files (begin with "version https://git-lfs.github.com/spec/v1")
            local sample="${npz_files[0]}"
            local header
            header="$(dd if="$sample" bs=1 count=64 2>/dev/null | tr -d '\000' || true)"
            if [[ "$header" == version\ https://git-lfs.github.com/spec/v1* ]]; then
                echo "[ERROR] Detected Git LFS pointer instead of real NPZ: $sample"
                echo "        Re-run download_assets.sh after ensuring hf login/network is OK."
                missing=1
            fi
        fi
    fi

    if [[ $missing -ne 0 ]]; then
        echo ""
        echo "[SIM1] Download finished, but folder layout is not readable by SIM1."
        echo "       Please check your Hugging Face credentials and rerun:"
        echo "       bash download_assets.sh"
        exit 1
    fi

    echo "[SIM1] Verified folder layout:"
    echo "       $DEST/acone/acone.urdf"
    echo "       $DEST/cloth/short-shirt.usdc"
    echo "       $DEST/model/flow_ckpt_three.pth"
    echo "       $DEST/$DATASET_SUBDIR/npz/*.npz"
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
if ! ensure_model_folder; then
    echo "[WARN] Could not auto-download model folder; will verify at final checks."
fi

if ! download_dataset_subset_hf; then
    echo "[SIM1] HF dataset subset download failed; trying git sparse-checkout + LFS ..."
    if ! download_dataset_subset_git; then
        echo "[ERROR] Failed to download dataset subset."
        echo "  Repo: https://huggingface.co/datasets/${DATASET_REPO}"
        echo "  Required folder: ${DATASET_SUBDIR}"
        echo "  If repo is private/gated, run: hf auth login"
        exit 1
    fi
fi

normalize_dataset_npz_layout
verify_final_layout

echo "[SIM1] Done. Assets and reference NPZ subset saved to: $DEST"
print_sim1_assets_hint
