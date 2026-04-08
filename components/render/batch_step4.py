#!/usr/bin/env python3
"""
Batch Step 4 only: for each record id under root_dir/npz/*.npz (sorted),
pick the newest root_dir/blend_out/<id>/*.blend and call step_04.

Sharding matches a simple [start_id * freq, (start_id+1) * freq) index range
over the sorted list (same idea as the Garment MeisterRender bash).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_RENDER = os.path.dirname(os.path.abspath(__file__))
if _RENDER not in sys.path:
    sys.path.insert(0, _RENDER)


def latest_blend(blend_out_subdir: str) -> str | None:
    paths = glob.glob(os.path.join(blend_out_subdir, "*.blend"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MeisterRender Step 4 (step4_render_acone.step_04)")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Session directory (contains npz/, camera/, blend_out/)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Fold the blue short shirt",
        help="Stored in meta_info.pkl",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Shard index: process global indices [start_id*freq, (start_id+1)*freq)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=10**9,
        help="Number of records per shard (default: all in one run)",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    npz_dir = os.path.join(root_dir, "npz")
    if not os.path.isdir(npz_dir):
        print(f"Error: missing npz dir: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    names = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    start = args.start_id * args.freq
    end = (args.start_id + 1) * args.freq

    print(f"[batch_step4] root_dir={root_dir}")
    print(f"[batch_step4] sorted npz count={len(names)}, index range [{start}, {end})")

    for idx, name in enumerate(names):
        record_id = os.path.splitext(name)[0]
        if idx < start or idx >= end:
            print(f"[{idx}] skip {record_id}")
            continue

        cam = os.path.join(root_dir, "camera", f"{record_id}_exts.npz")
        npz_path = os.path.join(root_dir, "npz", f"{record_id}.npz")
        sub = os.path.join(root_dir, "blend_out", record_id)
        blend_path = latest_blend(sub)

        if not os.path.isfile(cam):
            print(f"[{idx}] skip {record_id}: missing {cam}", file=sys.stderr)
            continue
        if not os.path.isfile(npz_path):
            print(f"[{idx}] skip {record_id}: missing {npz_path}", file=sys.stderr)
            continue
        if not blend_path:
            print(f"[{idx}] skip {record_id}: no .blend under {sub}", file=sys.stderr)
            continue

        print(f"[{idx}] step_04 {record_id} <- {blend_path}")
        from step4_render_acone import step_04

        step_04(args.language, record_id, blend_path, root_dir)

    print("[batch_step4] Done.")


if __name__ == "__main__":
    main()
