#!/usr/bin/env bash
# ============================================================
# Batch Step 4 only (MeisterRender / step4_render_acone.py)
#
# Run after: python components/render/main.py --root_dir <SESSION>
# (main defaults to Steps 1–3 only; use main --step4 to inline Step 4.)
#
# Usage:
#   bash components/render/batch_step4.sh <ROOT_DIR> [LANGUAGE] [START_ID] [FREQ]
#
# Examples:
#   bash components/render/batch_step4.sh ./replay/my_run_0001
#   bash components/render/batch_step4.sh ./replay/my_run_0001 "Fold the shirt" 0 4
#
# Sharding: same as Garment script — global index over sorted npz/*.npz;
#   worker k processes indices [k*freq, (k+1)*freq).
#
# Conda: use the env where MeisterRender deps are installed (e.g. sim1).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT_DIR="${1:?Usage: $0 <ROOT_DIR> [LANGUAGE] [START_ID] [FREQ]}"
LANGUAGE="${2:-Fold the blue short shirt}"
START_ID="${3:-0}"
FREQ="${4:-999999999}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/batch_step4.py" "${ROOT_DIR}" --language "${LANGUAGE}" --start_id "${START_ID}" --freq "${FREQ}"
