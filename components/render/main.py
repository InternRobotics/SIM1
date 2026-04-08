#!/usr/bin/env python3
import os
import argparse
import sys

# Add render module directory to path for local imports
_render_dir = os.path.dirname(os.path.abspath(__file__))
if _render_dir not in sys.path:
    sys.path.insert(0, _render_dir)

# Step modules import bpy; defer imports until after argparse so `--help` works without Blender.


def main():
    from asset_paths import validate_render_assets_or_exit

    parser = argparse.ArgumentParser(description="USD to Blend pipeline with sharding")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing 'npz' and 'usd' subfolders")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID (0 to num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (i.e., GPU tasks)")
    parser.add_argument(
        "--skip_asset_check",
        action="store_true",
        help="Do not verify assets/ before running (not recommended)",
    )
    parser.add_argument(
        "--step3",
        type=str,
        choices=("random", "no_random"),
        default="random",
        help=(
            "Step 3 scene script: 'random' = random HDRI/table/cloth "
            "(step3_randomize_scene_acone_random_texture); "
            "'no_random' = fixed scene (step3_randomize_scene_acone_no_random). "
            "Default: random."
        ),
    )
    parser.add_argument(
        "--step4",
        action="store_true",
        help=(
            "After Step 3, run Step 4 (MeisterRender) in-process. "
            "Default: Steps 1–3 only; use batch_step4.sh / batch_step4.py for batch Step 4."
        ),
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="Fold the blue short shirt",
        help="Language string stored in Step 4 meta_info.pkl",
    )
    args = parser.parse_args()

    if not args.skip_asset_check:
        validate_render_assets_or_exit()

    from step1_usd2blend import step_01
    from step2_compute_camera_acone_yourdf import step_02

    if args.step3 == "random":
        from step3_randomize_scene_acone_random_texture import step_03
    else:
        from step3_randomize_scene_acone_no_random import step_03

    print(f"[main] Step 3 mode: {args.step3} ({'randomize' if args.step3 == 'random' else 'deterministic'})")
    if args.step4:
        print("[main] Step 4: enabled (--step4), MeisterRender runs after each Step 3")
    else:
        print("[main] Step 4: off (default). Batch render: bash components/render/batch_step4.sh <ROOT_DIR>")

    npz_dir = os.path.join(args.root_dir, "npz")
    if not os.path.exists(npz_dir):
        print(f" Error: npz directory not found: {npz_dir}")
        sys.exit(1)

    # Collect all .npz files; sort so sharding is deterministic
    all_npz = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    total = len(all_npz)

    if total == 0:
        print(f"  Warning: No .npz files found in {npz_dir}")
        return

    # Shard: keep files where i % num_shards == shard_id
    my_files = [f for i, f in enumerate(all_npz) if i % args.num_shards == args.shard_id]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Total files: {total}, assigned: {len(my_files)}")

    for npz_filename in my_files:
        record_id = os.path.splitext(npz_filename)[0]
        usd_file = os.path.join(args.root_dir, "usd", f"{record_id}.usd")
        blend_dir = os.path.join(args.root_dir, "blend")
        out_blend_path = os.path.join(blend_dir, f"{record_id}.blend")

        if not os.path.exists(usd_file):
            print(f"  Skipping {record_id}: USD file not found")
            continue

        # Step 01: USD → Blend
        os.makedirs(blend_dir, exist_ok=True)
        step_01(usd_file, out_blend_path)

        # Step 02: Compute camera
        step_02(args.root_dir, record_id)

        # Step 03: Randomize scene & inject camera
        render_blend_file = step_03(out_blend_path, args.root_dir, record_id)

        if args.step4:
            from step4_render_acone import step_04

            step_04(args.language_instruction, record_id, render_blend_file, args.root_dir)

    print(f"[Shard {args.shard_id}]  Done.")


if __name__ == "__main__":
    main()