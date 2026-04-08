#!/usr/bin/env bash
# ============================================================
# Multi-GPU parallel launcher for main.py (Steps 1–3, optionally +Step 4)
#
# Scans npz/ and usd/ under ROOT_DIR, pairs them by stem name,
# then launches one main.py worker per GPU. Workers share the
# full file list via modulo sharding:
#   worker i  handles files where  index % NUM_GPUS == i
# This guarantees even distribution; uneven N/NUM_GPUS is fine
# (some workers get one extra file, none sit completely idle).
#
# Usage:
#   bash components/render/main_parallel.sh <ROOT_DIR> [NUM_GPUS] [EXTRA_ARGS...]
#
#   EXTRA_ARGS are forwarded verbatim to main.py, e.g.:
#     --step4  --language_instruction "Fold the shirt"  --step3 no_random
#
# Examples:
#   # Single GPU (equivalent to: python main.py --root_dir ./run)
#   bash components/render/main_parallel.sh ./replay/my_run 1
#
#   # 4 GPUs, Steps 1–3 only
#   bash components/render/main_parallel.sh ./replay/my_run 4
#
#   # 4 GPUs, Steps 1–4, with language instruction
#   bash components/render/main_parallel.sh ./replay/my_run 4 \
#       --step4 --language_instruction "Fold the shirt"
#
#   # 2 GPUs, fixed scene (no texture randomization)
#   bash components/render/main_parallel.sh ./replay/my_run 2 --step3 no_random
#
# Notes:
#   - GPU assignment: worker i gets CUDA_VISIBLE_DEVICES=i.
#     If your GPUs are not 0..N-1, export CUDA_VISIBLE_DEVICES before calling
#     or edit the assignment below.
#   - Steps 1–3 are CPU-bound (Blender). --step4 (MeisterRender) uses GPU.
#     For Steps 1–3 only you can safely exceed physical GPU count.
#   - Log files per worker: /tmp/sim1_worker_<i>.log
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Parse args ───────────────────────────────────────────────
ROOT_DIR="${1:?Usage: $0 <ROOT_DIR> [NUM_GPUS] [EXTRA_ARGS...]}"
NUM_GPUS="${2:-1}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "[main_parallel] Error: NUM_GPUS must be a positive integer, got: '${NUM_GPUS}'" >&2
    exit 1
fi

# ── Count valid (npz, usd) pairs ─────────────────────────────
NPZ_DIR="${ROOT_DIR}/npz"
USD_DIR="${ROOT_DIR}/usd"

if [[ ! -d "${NPZ_DIR}" ]]; then
    echo "[main_parallel] Error: npz dir not found: ${NPZ_DIR}" >&2
    exit 1
fi
if [[ ! -d "${USD_DIR}" ]]; then
    echo "[main_parallel] Error: usd dir not found: ${USD_DIR}" >&2
    exit 1
fi

# Count paired files (npz stem must have matching .usd)
N=0
while IFS= read -r npz_path; do
    stem="$(basename "${npz_path}" .npz)"
    [[ -f "${USD_DIR}/${stem}.usd" ]] && N=$((N + 1))
done < <(find "${NPZ_DIR}" -maxdepth 1 -name '*.npz' | sort)

if [[ "${N}" -eq 0 ]]; then
    echo "[main_parallel] Error: no matching (npz, usd) pairs found under ${ROOT_DIR}" >&2
    exit 1
fi

# Clamp workers to file count
if [[ "${NUM_GPUS}" -gt "${N}" ]]; then
    echo "[main_parallel] Warning: NUM_GPUS (${NUM_GPUS}) > pairs (${N}); clamping to ${N}"
    NUM_GPUS="${N}"
fi

# Compute per-worker file count (modulo sharding: ±1 difference max)
EACH_MIN=$(( N / NUM_GPUS ))
EACH_MAX=$(( EACH_MIN + 1 ))
EXTRA=$(( N % NUM_GPUS ))   # first EXTRA workers get EACH_MAX, rest get EACH_MIN

echo "============================================================"
echo "[main_parallel] root_dir  : ${ROOT_DIR}"
echo "[main_parallel] pairs     : ${N} (npz+usd matched)"
echo "[main_parallel] workers   : ${NUM_GPUS}"
if [[ "${EXTRA}" -gt 0 ]]; then
    echo "[main_parallel] distribution: ${EXTRA} workers × ${EACH_MAX} files, $((NUM_GPUS - EXTRA)) workers × ${EACH_MIN} files"
else
    echo "[main_parallel] distribution: all workers × ${EACH_MIN} files (even)"
fi
echo "[main_parallel] extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Launch workers ────────────────────────────────────────────
pids=()
for (( i = 0; i < NUM_GPUS; i++ )); do
    LOG="/tmp/sim1_worker_${i}.log"
    echo "[main_parallel] Launching worker ${i}/${NUM_GPUS} → GPU ${i}, log: ${LOG}"
    (
        export CUDA_VISIBLE_DEVICES="${i}"
        python "${SCRIPT_DIR}/main.py" \
            --root_dir "${ROOT_DIR}" \
            --shard_id "${i}" \
            --num_shards "${NUM_GPUS}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
            2>&1 | tee "${LOG}"
    ) &
    pids+=("$!")
done

echo "[main_parallel] All ${NUM_GPUS} worker(s) launched. Waiting for completion..."

# ── Collect results ───────────────────────────────────────────
ec=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[main_parallel] Worker (pid ${pid}) exited with error." >&2
        ec=1
    fi
done

echo "============================================================"
if [[ "${ec}" -ne 0 ]]; then
    echo "[main_parallel] FAILED — one or more workers reported errors." >&2
    echo "[main_parallel] Check /tmp/sim1_worker_*.log for details." >&2
    exit 1
fi
echo "[main_parallel] All done. ${N} trajectories across ${NUM_GPUS} GPU(s)."
echo "============================================================"
