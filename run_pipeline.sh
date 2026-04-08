#!/bin/bash
# ============================================================
# SIM1 data generation pipeline (run_pipeline.sh)
#
# Unified script that chains the full data-generation workflow:
#   1. Generate trajectories  (datagen_app.py)
#   2. Kalman-smooth           (smooth_trajectory_multi_thread.py)
#   3. Replay                  (replay_app.py)
#   4. Filter bad trajectories (rigid-transform aware)
#
# The user chooses whether to enable cloth position randomization at replay.
# The script automatically runs the matching
# filter pipeline:
#   - No randomization  →  cloth quality filter only
#   - With randomization →  EE reachability filter first,
#                           then cloth quality filter (aligned)
#
# Prerequisite (once): download SIM1 assets from Hugging Face into ./assets/
#   bash download_assets.sh
# Paths are resolved via SIM1_ASSETS_ROOT (default: <repo>/assets). Same variable as Python envs/render.
#
# Usage:
#   bash run_pipeline.sh [OPTIONS]
#
# Examples:
#   bash download_assets.sh          # once per machine / after clone
#   bash run_pipeline.sh --num 10  # generate 10 trajectories, no cloth position randomization
#
#   # With cloth position randomization at replay (±2cm XY, ±15° Z)
#   bash run_pipeline.sh --num 50 --position-randomize
#
#   # Custom data folder & session name
#   bash run_pipeline.sh --num 20 --data_folder ./dataset/my_task \
#                        --folder_name my_session --workers 16
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Defaults ──────────────────────────────────────────────────
# If --data_folder is not provided, auto-resolve to:
#   1) <SIM1_ASSETS_ROOT>/sim_teleoperated_npz (downloaded from HF dataset subset)
#   2) fallback: <repo>/dataset/example
DATA_FOLDER=""
REF_NPZ_FOLDER=""
NUM=10
WORKERS=8
FOLDER_NAME="pipeline_output"
POSITION_RANDOMIZE=false
REF_USD=""
SKIP_DATAGEN=false
SKIP_SMOOTH=false
SKIP_REPLAY=false
SKIP_FILTER=false
SKIP_ASSET_CHECK=false

# ── Resolve Hugging Face asset root (same rules as sim1_asset_paths.py) ──
resolve_assets_root() {
    if [[ -z "${SIM1_ASSETS_ROOT:-}" ]]; then
        echo "${PROJECT_ROOT}/assets"
        return
    fi
    if [[ "${SIM1_ASSETS_ROOT}" = /* ]]; then
        echo "${SIM1_ASSETS_ROOT}"
    else
        echo "${PROJECT_ROOT}/${SIM1_ASSETS_ROOT}"
    fi
}

# ── Resolve default reference-data folder for DataGen ─────────
resolve_default_data_folder() {
    local candidate="${SIM1_ASSETS_ROOT}/sim_teleoperated_npz"
    if [[ -d "${candidate}" ]]; then
        echo "${candidate}"
    else
        echo "${PROJECT_ROOT}/dataset/example"
    fi
}

resolve_npz_dir() {
    # Accept either:
    #   A) <root>/npz/*.npz
    #   B) <root>/*.npz
    local root="$1"
    local npz_sub="${root}/npz"
    if [[ -d "${npz_sub}" ]]; then
        echo "${npz_sub}"
        return 0
    fi
    shopt -s nullglob
    local direct_npz=("${root}"/*.npz)
    shopt -u nullglob
    if [[ ${#direct_npz[@]} -gt 0 ]]; then
        echo "${root}"
        return 0
    fi
    return 1
}

# ── Ensure DataGen-compatible layout: <data_folder>/npz/*.npz ─
ensure_datagen_npz_layout() {
    local root="$1"
    local npz_dir="${root}/npz"
    if [[ -d "${npz_dir}" ]]; then
        return 0
    fi

    shopt -s nullglob
    local direct_npz=("${root}"/*.npz)
    shopt -u nullglob
    if [[ ${#direct_npz[@]} -eq 0 ]]; then
        return 0
    fi

    mkdir -p "${npz_dir}"
    local f b
    for f in "${direct_npz[@]}"; do
        b="$(basename "$f")"
        ln -sfn "../${b}" "${npz_dir}/${b}"
    done
    echo "  Prepared reference view: ${npz_dir} (symlinks to ${root}/*.npz)"
}

verify_reference_npz() {
    # DataGen needs reference npz under <data_folder>/npz.
    if $SKIP_DATAGEN; then
        return 0
    fi

    local npz_dir="${DATA_FOLDER}/npz"
    if [[ ! -d "${npz_dir}" ]]; then
        echo "[ERROR] Reference NPZ directory not found: ${npz_dir}"
        echo "        Run: bash download_assets.sh"
        echo "        Expected HF subset: InternRobotics/Sim1_Dataset/sim_teleoperated_npz"
        exit 1
    fi

    shopt -s nullglob
    local files=("${npz_dir}"/*.npz)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "[ERROR] No .npz files found in: ${npz_dir}"
        echo "        Run: bash download_assets.sh"
        exit 1
    fi
}

prepare_reference_npz_link() {
    # If user provides --ref_npz_folder, link its npz source into <data_folder>/npz
    # so datagen keeps using the expected layout.
    if [[ -z "${REF_NPZ_FOLDER}" ]]; then
        return 0
    fi

    local src_root="$REF_NPZ_FOLDER"
    if [[ ! "$src_root" = /* ]]; then
        src_root="${PROJECT_ROOT}/${src_root}"
    fi
    if [[ ! -d "${src_root}" ]]; then
        echo "[ERROR] --ref_npz_folder does not exist: ${src_root}"
        exit 1
    fi

    local resolved_src=""
    if ! resolved_src="$(resolve_npz_dir "${src_root}")"; then
        echo "[ERROR] --ref_npz_folder has no .npz data: ${src_root}"
        echo "        Expected either <ref_npz_folder>/npz/*.npz or <ref_npz_folder>/*.npz"
        exit 1
    fi

    mkdir -p "${DATA_FOLDER}"
    local target_npz="${DATA_FOLDER}/npz"
    if [[ -e "${target_npz}" && ! -L "${target_npz}" ]]; then
        echo "[ERROR] ${target_npz} exists and is not a symlink."
        echo "        Please remove it or choose a different --data_folder."
        exit 1
    fi
    ln -sfn "${resolved_src}" "${target_npz}"
    echo "  Using reference NPZ from: ${resolved_src}"
}

# ── Verify SIM1 assets (URDF/cloth from HF bundle; see SIM1_ASSETS_ROOT) ──
verify_sim1_assets() {
    if $SKIP_ASSET_CHECK; then
        return 0
    fi
    local ASSETS_ROOT
    ASSETS_ROOT="$(resolve_assets_root)"
    export SIM1_ASSETS_ROOT="${ASSETS_ROOT}"
    local missing=0
    if [[ ! -f "${ASSETS_ROOT}/acone/acone.urdf" ]]; then
        echo "[ERROR] Missing: ${ASSETS_ROOT}/acone/acone.urdf"
        missing=1
    fi
    if [[ ! -f "${ASSETS_ROOT}/cloth/short-shirt.usdc" ]]; then
        echo "[ERROR] Missing: ${ASSETS_ROOT}/cloth/short-shirt.usdc"
        missing=1
    fi
    if [[ ! -f "${ASSETS_ROOT}/model/flow_ckpt_three.pth" ]]; then
        echo "[ERROR] Missing diffusion checkpoint: ${ASSETS_ROOT}/model/flow_ckpt_three.pth"
        missing=1
    fi
    if [[ $missing -ne 0 ]]; then
        echo ""
        echo "Download SIM1 assets from Hugging Face, then re-run:"
        echo "  bash download_assets.sh                    # default: <repo>/assets"
        echo "  # or set SIM1_ASSETS_ROOT to your --local-dir from hf download"
        echo ""
        exit 1
    fi
}

# ── Parse arguments ──────────────────────────────────────────
print_help() {
    cat <<'HELP'
SIM1 data generation pipeline

USAGE:
    bash run_pipeline.sh [OPTIONS]

OPTIONS:
  Data Generation:
    --data_folder DIR       Data root for generated outputs (gen/, gen/kf/).
                            If omitted:
                              - with --ref_npz_folder: <repo>/dataset/example
                              - otherwise: auto-detect
                                <SIM1_ASSETS_ROOT>/sim_teleoperated_npz
                                (fallback <repo>/dataset/example)
    --ref_npz_folder DIR    Reference NPZ source for DataGen.
                            Accepts either:
                              <DIR>/npz/*.npz
                              <DIR>/*.npz
                            If set, run_pipeline links it to <data_folder>/npz.
    --num N                 Number of trajectories to generate (default: 10)
    --skip_datagen          Skip trajectory generation (use existing data)

  Smoothing:
    --workers N             Parallel workers for smoothing & filtering (default: 8)
    --skip_smooth           Skip trajectory smoothing

  Replay:
    --folder_name NAME      Session base name under replay/ (default: pipeline_output)
    --position-randomize    Random cloth pose at replay (±2cm XY, ±15° yaw). Automatically
                            runs EE-reachability filter + aligned cloth-quality filter.
                            Omit → standard cloth-quality filter only.
    --skip_replay           Skip replay

  Filtering:
    --ref_usd PATH          Reference USD for Kabsch alignment (auto-detected if omitted)
    --skip_filter           Skip post-processing filters

  General:
    --skip_asset_check      Skip checking HF assets before run (experts only)
    --help                  Show this help
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_folder)       DATA_FOLDER="$2"; shift 2 ;;
        --ref_npz_folder)    REF_NPZ_FOLDER="$2"; shift 2 ;;
        --num)               NUM="$2"; shift 2 ;;
        --workers)           WORKERS="$2"; shift 2 ;;
        --folder_name)       FOLDER_NAME="$2"; shift 2 ;;
        --position-randomize) POSITION_RANDOMIZE=true; shift ;;
        --ref_usd)           REF_USD="$2"; shift 2 ;;
        --skip_datagen)      SKIP_DATAGEN=true; shift ;;
        --skip_smooth)       SKIP_SMOOTH=true; shift ;;
        --skip_replay)       SKIP_REPLAY=true; shift ;;
        --skip_filter)       SKIP_FILTER=true; shift ;;
        --skip_asset_check) SKIP_ASSET_CHECK=true; shift ;;
        --help|-h)           print_help ;;
        *) echo "Unknown option: $1"; print_help ;;
    esac
done

verify_sim1_assets

# Match Python sim1_asset_paths.py: subprocesses read the same HF bundle root
export SIM1_ASSETS_ROOT="$(resolve_assets_root)"

# Auto-select DataGen reference source when --data_folder is omitted.
if [[ -z "${DATA_FOLDER}" ]]; then
    if [[ -n "${REF_NPZ_FOLDER}" ]]; then
        DATA_FOLDER="${PROJECT_ROOT}/dataset/example"
    else
        DATA_FOLDER="$(resolve_default_data_folder)"
    fi
fi

# ── Resolve paths ────────────────────────────────────────────
if [[ ! "$DATA_FOLDER" = /* ]]; then
    DATA_FOLDER="${PROJECT_ROOT}/${DATA_FOLDER}"
fi

# Reference NPZ source:
#   - explicit --ref_npz_folder (preferred when provided)
#   - otherwise infer from --data_folder layout
prepare_reference_npz_link
if [[ -z "${REF_NPZ_FOLDER}" ]]; then
    # Accept both layouts:
    #   A) <data_folder>/npz/*.npz  (native)
    #   B) <data_folder>/*.npz      (HF subset folder root)
    ensure_datagen_npz_layout "$DATA_FOLDER"
fi
verify_reference_npz

GEN_DIR="${DATA_FOLDER}/gen"
SMOOTH_DIR="${GEN_DIR}/kf"

# ── Banner ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SIM1 data generation pipeline"
echo "============================================================"
echo "  HF assets    : $SIM1_ASSETS_ROOT  (override: export SIM1_ASSETS_ROOT=...)"
echo "  Data folder  : $DATA_FOLDER"
if [[ -n "${REF_NPZ_FOLDER}" ]]; then
echo "  Ref source   : $REF_NPZ_FOLDER"
fi
echo "  Ref NPZ dir  : ${DATA_FOLDER}/npz"
echo "  Trajectories : $NUM"
echo "  Position randomization (replay): $POSITION_RANDOMIZE"
if $POSITION_RANDOMIZE; then
echo "    (fixed ±2 cm XY, ±15° yaw on cloth at replay)"
fi
echo "  Workers      : $WORKERS"
echo "  Session name : $FOLDER_NAME"
echo "  GPU          : $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------"
echo "  Steps:"
$SKIP_DATAGEN && echo "    1. Generate    [SKIP]" || echo "    1. Generate    ✓"
$SKIP_SMOOTH  && echo "    2. Smooth      [SKIP]" || echo "    2. Smooth      ✓"
$SKIP_REPLAY  && echo "    3. Replay      [SKIP]" || echo "    3. Replay      ✓"
$SKIP_FILTER  && echo "    4. Filter      [SKIP]" || echo "    4. Filter      ✓  ($($POSITION_RANDOMIZE && echo 'EE reachability + cloth quality (aligned)' || echo 'cloth quality only'))"
echo "============================================================"
echo ""

# ── Step 1: Generate ─────────────────────────────────────────
if ! $SKIP_DATAGEN; then
    echo ">>> [1/4] Generating $NUM trajectories (diffusion-policy / fine split) ..."
    python "${PROJECT_ROOT}/apps/datagen_app.py" \
        --data_folder "$DATA_FOLDER" \
        --num "$NUM" \
        --use_dp \
        --mode fine
    echo ">>> [1/4] Generation complete. Output: ${GEN_DIR}"
    echo ""
fi

# ── Step 2: Smooth ───────────────────────────────────────────
if ! $SKIP_SMOOTH; then
    echo ">>> [2/4] Kalman-smoothing trajectories ..."
    python "${PROJECT_ROOT}/scripts/smooth_trajectory_multi_thread.py" \
        "$GEN_DIR" \
        "$SMOOTH_DIR" \
        --method kalman \
        --process_variance 8e-6 \
        --measurement_variance 2.5e-4 \
        --workers "$WORKERS"
    echo ">>> [2/4] Smoothing complete. Output: ${SMOOTH_DIR}"
    echo ""
fi

# ── Step 3: Replay ───────────────────────────────────────────
if ! $SKIP_REPLAY; then
    echo ">>> [3/4] Replaying smoothed trajectories ..."
    REPLAY_ARGS=("${PROJECT_ROOT}/apps/replay_app.py" "$SMOOTH_DIR" --folder_name "$FOLDER_NAME")
    if $POSITION_RANDOMIZE; then
        REPLAY_ARGS+=(--position-randomize)
    fi
    python "${REPLAY_ARGS[@]}"
    echo ">>> [3/4] Replay complete."
    echo ""
fi

# ── Detect the latest session directory ──────────────────────
REPLAY_ROOT="${PROJECT_ROOT}/replay"
LATEST_SESSION=""
if [[ -d "$REPLAY_ROOT" ]]; then
    LATEST_SESSION=$(ls -1d "${REPLAY_ROOT}/${FOLDER_NAME}_"[0-9][0-9][0-9][0-9] 2>/dev/null \
                     | sort | tail -n1 || true)
fi

if [[ -z "$LATEST_SESSION" ]] && ! $SKIP_FILTER; then
    echo "[ERROR] No session directory found under ${REPLAY_ROOT}/${FOLDER_NAME}_*"
    echo "        Run replay first or check --folder_name."
    exit 1
fi
echo "  Session dir  : $LATEST_SESSION"

# ── Step 4: Filter ───────────────────────────────────────────
if ! $SKIP_FILTER; then
    # Joint jump / first-frame mutation (always). EE FK check only with --position-randomize.
    echo ">>> [4/4] Filtering joint discontinuities (jump + first-5-frame mutation) ..."
    if $POSITION_RANDOMIZE; then
        python "${PROJECT_ROOT}/scripts/filter_joint_unreachable.py" \
            "${LATEST_SESSION}/npz" \
            --usd-dir "${LATEST_SESSION}/usd" \
            --workers "$WORKERS"
    else
        python "${PROJECT_ROOT}/scripts/filter_joint_unreachable.py" \
            "${LATEST_SESSION}/npz" \
            --usd-dir "${LATEST_SESSION}/usd" \
            --workers "$WORKERS" \
            --no-check-ee
    fi
    echo ""

    if $POSITION_RANDOMIZE; then
        # ── Cloth quality filter (aligned via Kabsch) ──
        echo ">>> [4/4] Filtering by cloth quality (aligned mode) ..."
        if [[ -n "$REF_USD" ]]; then
            REF_USD_PATH="$REF_USD"
        else
            REF_USD_PATH=$(ls -1 "${LATEST_SESSION}/usd/"*.usd 2>/dev/null | head -n1 || true)
            if [[ -z "$REF_USD_PATH" ]]; then
                echo "[WARN] No USD files found for --ref-usd auto-detection; skipping cloth filter."
            else
                echo "  Auto-detected ref USD: $REF_USD_PATH"
            fi
        fi
        if [[ -n "$REF_USD_PATH" ]]; then
            python "${PROJECT_ROOT}/scripts/filter_cloth_quality.py" \
                "$LATEST_SESSION" \
                --ref-usd "$REF_USD_PATH"
        fi
    else
        # ── 4: Cloth quality filter (direct mode) ──
        echo ">>> [4/4] Filtering by cloth quality (direct mode) ..."
        python "${PROJECT_ROOT}/scripts/filter_cloth_quality.py" \
            "$LATEST_SESSION"
    fi
    echo ">>> [4/4] Filtering complete."
    echo ""
fi

# ── Summary ──────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline finished!"
echo "============================================================"
echo "  Session       : $LATEST_SESSION"
if [[ -d "${LATEST_SESSION}/npz" ]]; then
    NPZ_COUNT=$(ls -1 "${LATEST_SESSION}/npz/"*.npz 2>/dev/null | wc -l || echo 0)
    echo "  Good trajs    : ${NPZ_COUNT} .npz files"
fi
echo ""
echo "  Next steps:"
echo "    # Render (Steps 1-3 + Step 4):"
echo "    python components/render/main.py --root_dir ${LATEST_SESSION}"
echo "    bash components/render/batch_step4.sh ${LATEST_SESSION}"
echo ""
echo "    # Convert to LeRobot dataset:"
echo "    bash components/lmdb2lerobot/run_local.sh \\"
echo "      --src ${LATEST_SESSION}/out_updated --out ${LATEST_SESSION}/lerobot_dataset"
echo "============================================================"
