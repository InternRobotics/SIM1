#!/usr/bin/env bash
# ============================================================
# Multi-GPU parallel launcher for Step 4 only (MeisterRender).
#
# Run this AFTER main.py (or main_parallel.sh) has produced
# npz/, camera/, blend_out/ under ROOT_DIR.
#
# Sharding strategy: ceil(N / NUM_WORKERS) files per worker.
#   Worker i processes global index range  [i*FREQ, (i+1)*FREQ).
#   Uneven N/NUM_WORKERS is handled automatically — the last
#   worker simply has fewer items (no idle workers).
#
# Usage:
#   bash components/render/batch_step4_parallel.sh \
#       <ROOT_DIR> [LANGUAGE] [NUM_WORKERS] [BASE_GPU]
#
#   ROOT_DIR    Session directory (contains npz/, camera/, blend_out/)
#   LANGUAGE    Language instruction stored in metadata  (default: "Fold the blue short shirt")
#   NUM_WORKERS Number of parallel processes             (default: 1)
#   BASE_GPU    First GPU index                          (default: 0)
#               Worker i uses CUDA_VISIBLE_DEVICES = BASE_GPU + i
#
# Examples:
#   # Single GPU — all trajectories, one after another
#   bash components/render/batch_step4_parallel.sh ./replay/my_run
#
#   # 4 GPUs (GPU 0–3), even split
#   bash components/render/batch_step4_parallel.sh ./replay/my_run "Fold the shirt" 4 0
#
#   # 2 GPUs starting from GPU 2 (GPU 2, GPU 3)
#   bash components/render/batch_step4_parallel.sh ./replay/my_run "Fold the shirt" 2 2
#
# Multi-machine: on each machine run batch_step4.sh with the same FREQ
# but different START_ID. Example for 3 machines each handling ~1/3:
#   Machine A: bash batch_step4.sh ROOT LANG 0 $(( (N+2)/3 ))
#   Machine B: bash batch_step4.sh ROOT LANG 1 $(( (N+2)/3 ))
#   Machine C: bash batch_step4.sh ROOT LANG 2 $(( (N+2)/3 ))
#
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT_DIR="${1:?Usage: $0 <ROOT_DIR> [LANGUAGE] [NUM_WORKERS] [BASE_GPU]}"
LANGUAGE="${2:-Fold the blue short shirt}"
NUM_WORKERS="${3:-1}"
BASE_GPU="${4:-0}"

if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || [[ "${NUM_WORKERS}" -lt 1 ]]; then
    echo "[batch_step4_parallel] Error: NUM_WORKERS must be >= 1, got: '${NUM_WORKERS}'" >&2
    exit 1
fi

NPZ_DIR="${ROOT_DIR}/npz"
if [[ ! -d "${NPZ_DIR}" ]]; then
    echo "[batch_step4_parallel] Error: npz dir not found: ${NPZ_DIR}" >&2
    exit 1
fi

mapfile -t _npzs < <(find "${NPZ_DIR}" -maxdepth 1 -name '*.npz' | sort)
N="${#_npzs[@]}"
if [[ "${N}" -eq 0 ]]; then
    echo "[batch_step4_parallel] Error: no .npz files under ${NPZ_DIR}" >&2
    exit 1
fi

# Clamp workers to file count (no point in idle workers)
if [[ "${NUM_WORKERS}" -gt "${N}" ]]; then
    echo "[batch_step4_parallel] Warning: NUM_WORKERS (${NUM_WORKERS}) > files (${N}); clamping to ${N}"
    NUM_WORKERS="${N}"
fi

# FREQ = ceil(N / NUM_WORKERS) — each worker's shard size
FREQ=$(( (N + NUM_WORKERS - 1) / NUM_WORKERS ))

# Per-worker count (last worker may have fewer)
LAST_WORKER_FILES=$(( N - (NUM_WORKERS - 1) * FREQ ))
[[ "${LAST_WORKER_FILES}" -lt 0 ]] && LAST_WORKER_FILES=0

echo "============================================================"
echo "[batch_step4_parallel] root_dir   : ${ROOT_DIR}"
echo "[batch_step4_parallel] language   : ${LANGUAGE}"
echo "[batch_step4_parallel] npz files  : ${N}"
echo "[batch_step4_parallel] workers    : ${NUM_WORKERS}"
echo "[batch_step4_parallel] shard size : ${FREQ} (last worker: ${LAST_WORKER_FILES})"
echo "[batch_step4_parallel] GPUs       : ${BASE_GPU}..$(( BASE_GPU + NUM_WORKERS - 1 ))"
echo "============================================================"

cd "${PROJECT_ROOT}"

pids=()
for (( i = 0; i < NUM_WORKERS; i++ )); do
    START=$(( i * FREQ ))
    # Skip if this worker has no files (shouldn't happen after clamping, but be safe)
    [[ "${START}" -ge "${N}" ]] && break

    GPU=$(( BASE_GPU + i ))
    echo "[batch_step4_parallel] Worker ${i}: indices [${START}, $(( START + FREQ ))), GPU ${GPU}"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        bash "${SCRIPT_DIR}/batch_step4.sh" \
            "${ROOT_DIR}" "${LANGUAGE}" "${i}" "${FREQ}"
    ) 2>&1 | sed -u "s/^/[W${i}|GPU${GPU}] /" &
    pids+=("$!")
done

echo "[batch_step4_parallel] ${#pids[@]} worker(s) running. Waiting..."

ec=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[batch_step4_parallel] Worker (pid ${pid}) exited with error." >&2
        ec=1
    fi
done

echo "============================================================"
if [[ "${ec}" -ne 0 ]]; then
    echo "[batch_step4_parallel] FAILED — one or more workers exited with error." >&2
    exit 1
fi
echo "[batch_step4_parallel] All done. ${N} trajectories rendered across ${NUM_WORKERS} GPU(s)."
echo "============================================================"
