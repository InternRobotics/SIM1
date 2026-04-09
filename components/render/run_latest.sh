#!/usr/bin/env bash
# One-click render for latest replay session.
# Default session pattern: replay/pipeline_output_XXXX

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION_PREFIX="pipeline_output"

usage() {
  cat <<'EOF'
Usage:
  bash components/render/run_latest.sh [--session-prefix NAME]

Examples:
  bash components/render/run_latest.sh
  bash components/render/run_latest.sh --session-prefix my_run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-prefix)
      SESSION_PREFIX="${2:?}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

shopt -s nullglob
sessions=( "${REPO_ROOT}/replay/${SESSION_PREFIX}_"[0-9][0-9][0-9][0-9] )
shopt -u nullglob
if [[ ${#sessions[@]} -eq 0 ]]; then
  echo "Error: cannot find replay session under ${REPO_ROOT}/replay/${SESSION_PREFIX}_*" >&2
  echo "Run run_pipeline.sh first or set --session-prefix." >&2
  exit 1
fi

IFS=$'\n' sessions_sorted=($(printf '%s\n' "${sessions[@]}" | sort))
LATEST_SESSION="${sessions_sorted[-1]}"

echo "Using latest session: ${LATEST_SESSION}"
python "${SCRIPT_DIR}/main.py" --root_dir "${LATEST_SESSION}"
bash "${SCRIPT_DIR}/batch_step4.sh" "${LATEST_SESSION}"

echo "Render complete:"
echo "  ${LATEST_SESSION}/out_updated"
