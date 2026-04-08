#!/bin/bash
# Slurm cluster example (internal paths + srun).
# For local use, run instead:
#   bash components/lmdb2lerobot/run_local.sh [options]
# See components/lmdb2lerobot/README.md for full documentation.
set -euo pipefail

NAME="scale_mimicgen_nodet_dpv2_clean_smooth_noik_nojump_new_0324_clean"
SCRIPT_DIR="/mnt/petrelfs/zhouyunsong/zhouys/tools/lmdb2lerobot"
SRC_PATH="/mnt/petrelfs/zhouyunsong/tmp/ebench_t/arx_lift2/scale/${NAME}"
SAVE_PATH="/mnt/petrelfs/zhouyunsong/tmp/ebench_t/arx_lift2/transformed_data/Cloths/${NAME}"

cd "${SCRIPT_DIR}"
[[ ! -d "${SRC_PATH}" ]] && { echo "Error: source directory not found: ${SRC_PATH}"; exit 1; }

echo "=== Step 1: lmdb2lerobot_arx_sim ==="
srun -p ebench_t --gres=gpu:8 --cpus-per-task 128 \
    python lmdb2lerobot_arx_sim.py \
    --src_path "${SRC_PATH}" \
    --save_path "${SAVE_PATH}" \
    --origin_fps 60 \
    --target_fps 60

echo "=== Step 2: sim2real ==="
srun -p ebench_t \
    python sim2real.py --dir "${SAVE_PATH}"
