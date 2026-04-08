#!/usr/bin/env bash
# Convert LMDB -> LeRobot -> sim2real for a single source directory.
# Paths are specified via --src / --out. See README.md or run with --help.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="$(pwd -P)"

usage() {
  local code="${1:-0}"
  cat <<'EOF'
Usage:
  bash run_local.sh --src <input_dir> --out <output_dir> [options]

Required:
  --src    LMDB parent directory (e.g. .../out_updated) or a single episode
           directory (e.g. .../out_updated/000000)
  --out    LeRobot output root (deleted and recreated if it already exists;
           parent directory must exist)

Options:
  --skip-sim2real   skip sim2real.py post-processing
  --debug           process only the first episode (smoke test)
  --repo-id NAME    dataset repo_id written to metadata (default: arx_sim_local)
  --origin-fps N    source LMDB frame rate (default: 60)
  --target-fps N    output frame rate, must divide origin-fps (default: 60)
  --num-threads N   parallel LMDB read workers (default: 4)

Example:
  bash run_local.sh --src ./replay/my_run_0001/out_updated --out ./replay/my_run_0001/lerobot_dataset
EOF
  exit "${code}"
}

to_abs() {
  local p="$1"
  if [[ "${p}" == /* ]]; then
    printf '%s' "${p}"
  else
    printf '%s' "${INVOCATION_DIR}/${p}"
  fi
}

SRC=""
SAVE=""
SKIP_SIM2REAL=0
DEBUG_FLAG=()
REPO_ID="arx_sim_local"
ORIGIN_FPS=60
TARGET_FPS=60
NUM_THREADS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC="${2:?}"
      shift 2
      ;;
    --out)
      SAVE="${2:?}"
      shift 2
      ;;
    --skip-sim2real)
      SKIP_SIM2REAL=1
      shift
      ;;
    --debug)
      DEBUG_FLAG=(--debug)
      shift
      ;;
    --repo-id)
      REPO_ID="${2:?}"
      shift 2
      ;;
    --origin-fps)
      ORIGIN_FPS="${2:?}"
      shift 2
      ;;
    --target-fps)
      TARGET_FPS="${2:?}"
      shift 2
      ;;
    --num-threads)
      NUM_THREADS="${2:?}"
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

[[ -n "${SRC}" ]]  || { echo "Missing --src" >&2; usage 1; }
[[ -n "${SAVE}" ]] || { echo "Missing --out" >&2; usage 1; }

SRC="$(to_abs "${SRC}")"
SAVE="$(to_abs "${SAVE}")"

if [[ ! -d "${SRC}" ]]; then
  echo "Error: --src is not an existing directory: ${SRC}" >&2
  exit 1
fi
SRC="$(cd "${SRC}" && pwd)"

parent_save="$(dirname "${SAVE}")"
if [[ ! -d "${parent_save}" ]]; then
  echo "Error: parent of --out does not exist, please create it first: ${parent_save}" >&2
  exit 1
fi
SAVE="$(cd "${parent_save}" && pwd)/$(basename "${SAVE}")"

cd "${REPO_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Step 1: LMDB -> LeRobot ==="
echo "  src  : ${SRC}"
echo "  out  : ${SAVE}"
python "${SCRIPT_DIR}/lmdb2lerobot_arx_sim.py" \
  --src_path "${SRC}" \
  --save_path "${SAVE}" \
  --repo_id "${REPO_ID}" \
  --origin_fps "${ORIGIN_FPS}" \
  --target_fps "${TARGET_FPS}" \
  --num-threads "${NUM_THREADS}" \
  "${DEBUG_FLAG[@]}"

if [[ "${SKIP_SIM2REAL}" -eq 0 ]]; then
  echo "=== Step 2: sim2real (parquet) ==="
  python "${SCRIPT_DIR}/sim2real.py" --dir "${SAVE}"
else
  echo "=== Skipped sim2real (--skip-sim2real) ==="
fi

echo "Done. LeRobot dataset: ${SAVE}"
