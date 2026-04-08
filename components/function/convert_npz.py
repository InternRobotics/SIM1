#!/usr/bin/env python3
"""
Convert NPZ files from 21D joint_q to 19D joint_q format.

Usage:
    python components/function/convert_npz.py <src_folder> <dst_folder>
"""

import os
import argparse
import numpy as np
from pathlib import Path


def convert_npz_21_to_19(src_folder, dst_folder):
    """Convert NPZ files from 21D joint_q to 19D format."""
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    for npz_file in src_folder.glob("*.npz"):
        print(f"Processing {npz_file.name}...")
        data = np.load(npz_file)

        if 'joint_q' not in data:
            print(f"  Warning: No 'joint_q', skipping.")
            continue

        jq = data['joint_q']
        if jq.ndim == 1:
            T = 1
            jq = jq.reshape(1, -1)
        elif jq.ndim == 2:
            T = jq.shape[0]
        else:
            print(f"  Warning: Unexpected joint_q ndim: {jq.ndim}")
            continue

        if jq.shape[1] != 21:
            print(f"  Warning: Not 21D: {jq.shape}, skipping.")
            continue

        # Extract arrays
        arm14 = jq[:, :14]      # left7 + right7
        gripper2 = jq[:, 14:16] # left_gripper, right_gripper
        base3 = np.zeros((T, 3), dtype=jq.dtype)

        # Concatenate to 19D: [base3, arm14, gripper2]
        new_joint_q = np.concatenate([base3, arm14, gripper2], axis=1)  # (T, 19)

        # Restore to 1D if originally scalar
        if T == 1:
            new_joint_q = new_joint_q[0]

        # Save while preserving other fields
        new_data = {k: v for k, v in data.items()}
        new_data['joint_q'] = new_joint_q

        save_path = dst_folder / npz_file.name
        np.savez_compressed(save_path, **new_data)

    print("All files converted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ files from 21D to 19D joint_q format")
    parser.add_argument("src_folder", type=str, help="Source folder containing 21D NPZ files")
    parser.add_argument("dst_folder", type=str, help="Destination folder for converted 19D NPZ files")
    args = parser.parse_args()

    convert_npz_21_to_19(args.src_folder, args.dst_folder)
