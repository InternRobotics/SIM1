#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

GRIPPER_MIN = 0.001
GRIPPER_MAX = 0.044
GRIPPER_RANGE = GRIPPER_MAX - GRIPPER_MIN
SCALE_FACTOR = -3.24

# Column names in the data parquet produced by lmdb2lerobot_arx_sim (see FEATURES dict)
_RAW_STATE_ACTION_KEYS = (
    "states.left_joint.position",
    "states.left_gripper.position",
    "states.right_joint.position",
    "states.right_gripper.position",
    "actions.left_joint.position",
    "actions.left_gripper.position",
    "actions.right_joint.position",
    "actions.right_gripper.position",
)


def normalize_gripper(x):
    return SCALE_FACTOR * (x - GRIPPER_MIN) / GRIPPER_RANGE


def process_parquet(path: Path) -> None:
    df = pd.read_parquet(path)
    cols = set(df.columns)

    has_raw = all(k in cols for k in _RAW_STATE_ACTION_KEYS)
    has_packed = "observation.state" in cols and "action" in cols

    if not has_raw:
        if has_packed:
            print(f"[SKIP] already in sim2real format (observation.state / action present): {path}")
        else:
            print(f"[SKIP] not a frame parquet (no ARX states/actions columns), likely meta/tasks: {path}")
        return

    obs_list = []
    act_list = []

    for _, row in df.iterrows():
        # observation.state (14,)
        obs = np.concatenate([
            np.asarray(row["states.left_joint.position"], dtype=np.float32),
            np.asarray([normalize_gripper(row["states.left_gripper.position"])], dtype=np.float32),
            np.asarray(row["states.right_joint.position"], dtype=np.float32),
            np.asarray([normalize_gripper(row["states.right_gripper.position"])], dtype=np.float32),
        ])

        # action (14,)
        act = np.concatenate([
            np.asarray(row["actions.left_joint.position"], dtype=np.float32),
            np.asarray([normalize_gripper(row["actions.left_gripper.position"])], dtype=np.float32),
            np.asarray(row["actions.right_joint.position"], dtype=np.float32),
            np.asarray([normalize_gripper(row["actions.right_gripper.position"])], dtype=np.float32),
        ])

        obs_list.append(obs)
        act_list.append(act)

    new_df = pd.DataFrame({
        "observation.state": obs_list,
        "action": act_list,
    })

    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if col in df.columns:
            new_df[col] = df[col]

    new_df.to_parquet(path, index=False)
    print(f"[OVERWRITE] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing parquet files (searched recursively).",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    parquet_files = sorted(base_dir.rglob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet file(s)")

    for p in parquet_files:
        process_parquet(p)


if __name__ == "__main__":
    main()
