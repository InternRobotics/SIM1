#!/usr/bin/env bash
# Batch conversion: multiple LMDB directories -> LeRobot, with multi-process parallelism (optional GPU assignment).
#
# Usage A -- auto-scan parent directory (recommended):
#   bash run_batch.sh --scan /data/runs --out-root /output --workers 4
#
#   Every sub-directory that contains an out_updated/ folder (or lmdb/data.mdb directly)
#   becomes an independent job processed in parallel.
#
# Usage B -- explicit list file (one src path per line):
#   bash run_batch.sh --list jobs.txt --out-root /output --workers 4
#
# jobs.txt example:
#   /data/run_001/out_updated
#   /data/run_002/out_updated
#
# Options:
#   --workers N        number of parallel workers (default: 4)
#   --gpus 0,1,2,3    comma-separated GPU ids, assigned round-robin to workers;
#                      omit to leave CUDA_VISIBLE_DEVICES unset
#   --target-fps N     default 60
#   --origin-fps N     default 60
#   --repo-id NAME     default arx_sim_local
#   --num-threads N    LMDB read threads per worker (default: 4)
#   --skip-sim2real    skip sim2real.py post-processing
#   --dry-run          print task list without executing

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────
SCAN_DIR=""
LIST_FILE=""
OUT_ROOT=""
WORKERS=4
GPUS=""
TARGET_FPS=60
ORIGIN_FPS=60
REPO_ID="arx_sim_local"
NUM_THREADS=4
SKIP_SIM2REAL=0
DRY_RUN=0

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scan)          SCAN_DIR="${2:?}"; shift 2 ;;
    --list)          LIST_FILE="${2:?}"; shift 2 ;;
    --out-root)      OUT_ROOT="${2:?}"; shift 2 ;;
    --workers)       WORKERS="${2:?}"; shift 2 ;;
    --gpus)          GPUS="${2:?}"; shift 2 ;;
    --target-fps)    TARGET_FPS="${2:?}"; shift 2 ;;
    --origin-fps)    ORIGIN_FPS="${2:?}"; shift 2 ;;
    --repo-id)       REPO_ID="${2:?}"; shift 2 ;;
    --num-threads)   NUM_THREADS="${2:?}"; shift 2 ;;
    --skip-sim2real) SKIP_SIM2REAL=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "${OUT_ROOT}" ]]                        || { echo "Missing --out-root" >&2; usage 1; }
[[ -n "${SCAN_DIR}" || -n "${LIST_FILE}" ]]   || { echo "Missing --scan or --list" >&2; usage 1; }

mkdir -p "${OUT_ROOT}"
OUT_ROOT="$(cd "${OUT_ROOT}" && pwd)"

# ── collect source paths ──────────────────────────────────────────────────────
declare -a SRC_LIST=()

if [[ -n "${LIST_FILE}" ]]; then
  [[ -f "${LIST_FILE}" ]] || { echo "List file not found: ${LIST_FILE}" >&2; exit 1; }
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"              # strip inline comments
    line="${line//[[:space:]]/}"    # strip whitespace
    [[ -z "${line}" ]] && continue
    SRC_LIST+=("${line}")
  done < "${LIST_FILE}"
fi

if [[ -n "${SCAN_DIR}" ]]; then
  [[ -d "${SCAN_DIR}" ]] || { echo "Scan directory not found: ${SCAN_DIR}" >&2; exit 1; }
  SCAN_DIR="$(cd "${SCAN_DIR}" && pwd)"

  # Priority 1: .../out_updated (standard Step-4 output layout)
  while IFS= read -r -d '' d; do
    SRC_LIST+=("${d}")
  done < <(find "${SCAN_DIR}" -maxdepth 2 -type d -name "out_updated" -print0 2>/dev/null | sort -z)

  if [[ ${#SRC_LIST[@]} -eq 0 ]]; then
    # Priority 2: sub-directories that directly contain lmdb/data.mdb
    while IFS= read -r -d '' d; do
      SRC_LIST+=("$(dirname "${d}")")
    done < <(find "${SCAN_DIR}" -maxdepth 3 -name "data.mdb" -print0 2>/dev/null | sort -z)
    # deduplicate
    mapfile -t SRC_LIST < <(printf '%s\n' "${SRC_LIST[@]}" | awk '!seen[$0]++')
  fi
fi

if [[ ${#SRC_LIST[@]} -eq 0 ]]; then
  echo "No jobs found, exiting." >&2
  exit 1
fi

echo "Found ${#SRC_LIST[@]} job(s)"
echo "Output root : ${OUT_ROOT}"
echo "Workers     : ${WORKERS}"
[[ -n "${GPUS}" ]] && echo "GPUs        : ${GPUS}"
echo ""

# parse GPU list into array
declare -a GPU_ARR=()
if [[ -n "${GPUS}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${GPUS}"
fi

# ── per-job function (runs in a subshell) ─────────────────────────────────────
run_one() {
  local src="$1"
  local out="$2"
  local gpu_id="$3"   # empty string or a single GPU index

  local log_file="${out}.log"
  mkdir -p "$(dirname "${log_file}")"

  local cmd=(
    bash "${SCRIPT_DIR}/run_local.sh"
    --src "${src}"
    --out "${out}"
    --repo-id "${REPO_ID}"
    --origin-fps "${ORIGIN_FPS}"
    --target-fps "${TARGET_FPS}"
    --num-threads "${NUM_THREADS}"
  )
  [[ "${SKIP_SIM2REAL}" -eq 1 ]] && cmd+=(--skip-sim2real)

  [[ -n "${gpu_id}" ]] && export CUDA_VISIBLE_DEVICES="${gpu_id}"

  echo "[$(date '+%H:%M:%S')] START $(basename "${src}") -> ${out}" | tee -a "${log_file}"
  if "${cmd[@]}" >> "${log_file}" 2>&1; then
    echo "[$(date '+%H:%M:%S')] DONE  $(basename "${src}")" | tee -a "${log_file}"
    return 0
  else
    echo "[$(date '+%H:%M:%S')] FAIL  $(basename "${src}") (see ${log_file})" >&2
    return 1
  fi
}
export -f run_one
export SCRIPT_DIR REPO_ID ORIGIN_FPS TARGET_FPS NUM_THREADS SKIP_SIM2REAL

# ── semaphore-style scheduler ─────────────────────────────────────────────────
PIDS=()
TASK_IDX=0
FAIL_COUNT=0

for src in "${SRC_LIST[@]}"; do
  src_name="$(basename "${src}")"
  out="${OUT_ROOT}/${src_name}"
  gpu_id=""
  if [[ ${#GPU_ARR[@]} -gt 0 ]]; then
    gpu_id="${GPU_ARR[$((TASK_IDX % ${#GPU_ARR[@]}))]}"
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] src=${src}  out=${out}  gpu=${gpu_id:-auto}"
    TASK_IDX=$((TASK_IDX + 1))
    continue
  fi

  # wait for a free slot
  while [[ ${#PIDS[@]} -ge "${WORKERS}" ]]; do
    new_pids=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        new_pids+=("${pid}")
      else
        wait "${pid}" || FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
    done
    PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    [[ ${#PIDS[@]} -ge "${WORKERS}" ]] && sleep 2
  done

  run_one "${src}" "${out}" "${gpu_id}" &
  PIDS+=($!)
  TASK_IDX=$((TASK_IDX + 1))
done

# drain remaining jobs
for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
  wait "${pid}" || FAIL_COUNT=$((FAIL_COUNT + 1))
done

echo ""
echo "==============================="
echo "All done | total: ${#SRC_LIST[@]} | failed: ${FAIL_COUNT}"
echo "Output: ${OUT_ROOT}"
echo "==============================="
[[ "${FAIL_COUNT}" -eq 0 ]] || exit 1
