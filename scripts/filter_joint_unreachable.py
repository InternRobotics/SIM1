#!/usr/bin/env python3
"""
Multi-PROCESS filter for .npz trajectories:

  1) Joint discontinuities — always (when this script runs):
     - any adjacent frame with |Δq| > threshold over the full sequence ("Joint jump")
     - any adjacent pair in the first 5 frames above threshold ("Joint mutation (first 5 frames)")

  2) End-effector reachability (optional) — forward kinematics, same as before:
     - left/right EE x must not go below the initial frame’s x (useful after rigid replay
       augmentation). Disable with --no-check-ee.

Moves bad .npz AND corresponding .usd files to <input_dir>_unreachable/.

Uses process isolation (spawn) to avoid Warp/Newton global state conflicts.
"""

import os
import sys
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sim1_asset_paths import get_acone_urdf_path
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp


mp.set_start_method("spawn", force=True)


left_ee_body_names = {"left_link16"}
right_ee_body_names = {"right_link26"}

# Default: rad per frame; override with --jump-threshold
JOINT_JUMP_THRESHOLD = 0.5


def init_robot_fk(device="cpu"):
    """
    Initialize RobotFK in isolated process context.
    MUST be called inside each worker process AFTER fork/spawn.
    """
    import warp as wp
    import newton

    class RobotFK:
        def __init__(self, device="cpu"):
            self.device = device
            self._build_model()
            self._identify_ee_indices()
            self._init_state()

        def _build_model(self):
            builder = newton.ModelBuilder()
            urdf_path = get_acone_urdf_path()
            xform_base = wp.transform(p=wp.vec3(0.0, 0.0, 0.17))
            builder.add_urdf(
                urdf_path,
                floating=False,
                enable_self_collisions=False,
                xform=xform_base,
            )
            self.model = builder.finalize(requires_grad=False, device=self.device)

        def _identify_ee_indices(self):
            lee_idx = None
            ree_idx = None
            for i, key in enumerate(self.model.body_key):
                if key in left_ee_body_names:
                    lee_idx = i
                if key in right_ee_body_names:
                    ree_idx = i
            assert lee_idx is not None and ree_idx is not None, "EE bodies not found"
            self.lee_index = lee_idx
            self.ree_index = ree_idx

        def _init_state(self):
            self.state = self.model.state()
            import warp as wp

            self.joint_qd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)

        def joint_to_ee_pos(self, joint_q_seq):
            import warp as wp
            import newton

            device = self.device
            model = self.model
            state = self.state
            joint_q_wp = wp.array(joint_q_seq, dtype=wp.float32, device=device)
            T = joint_q_seq.shape[0]

            left_pos = np.empty((T, 3), dtype=np.float32)
            right_pos = np.empty((T, 3), dtype=np.float32)

            for t in range(T):
                q_t = joint_q_wp[t]
                newton.eval_fk(model, q_t, self.joint_qd, state)
                body_q = state.body_q.numpy()
                left_pos[t] = body_q[self.lee_index, :3]
                right_pos[t] = body_q[self.ree_index, :3]

            return left_pos, right_pos

    return RobotFK(device=device)


def has_joint_jump(joint_q_seq, threshold):
    """True if any adjacent frame has max |Δq| > threshold over the whole trajectory."""
    if len(joint_q_seq) < 2:
        return False
    diff = np.abs(joint_q_seq[1:] - joint_q_seq[:-1])
    return np.max(diff) > threshold


def has_initial_mutation(joint_q_seq, threshold):
    """
    True if any adjacent pair in the first 5 frames exceeds threshold.
    (If trajectory has fewer than 5 frames, all available frames are checked.)
    """
    if len(joint_q_seq) < 2:
        return False
    max_frames = min(5, len(joint_q_seq))
    for t in range(max_frames - 1):
        diff = np.abs(joint_q_seq[t + 1] - joint_q_seq[t])
        if np.max(diff) > threshold:
            return True
    return False


def _joint_issue_reasons(joint_q_seq, threshold):
    reasons = []
    if has_joint_jump(joint_q_seq, threshold=threshold):
        reasons.append("Joint jump")
    if has_initial_mutation(joint_q_seq, threshold=threshold):
        reasons.append("Joint mutation (first 5 frames)")
    return reasons


def worker_process(task_args):
    """
    Single-process worker: NO shared state with other processes.
    Returns list of results for its file chunk.
    """
    (file_chunk, npz_dir, joint_jump_threshold, check_ee, worker_id) = task_args

    fk = init_robot_fk(device="cpu")

    results = []
    for fname in file_chunk:
        npz_path = os.path.join(npz_dir, fname)
        result = {
            "fname": fname,
            "reasons": [],
            "needs_move": False,
            "error": None,
        }

        try:
            data = np.load(npz_path)
            if "joint_q" not in data:
                continue
            joint_q_seq = data["joint_q"]
            if joint_q_seq.ndim != 2 or joint_q_seq.shape[1] != fk.model.joint_dof_count:
                continue

            joint_reasons = _joint_issue_reasons(joint_q_seq, joint_jump_threshold)
            result["reasons"].extend(joint_reasons)

            if check_ee:
                left_pos, right_pos = fk.joint_to_ee_pos(joint_q_seq)
                init_left_x = left_pos[0, 0]
                init_right_x = right_pos[0, 0]
                if np.any(left_pos[:, 0] < init_left_x) or np.any(right_pos[:, 0] < init_right_x):
                    result["reasons"].append("EE unreachable")

            result["needs_move"] = bool(result["reasons"])
            results.append(result)

        except Exception as e:
            result["error"] = str(e)
            results.append(result)
            continue

    return results


def main(npz_dir, usd_dir=None, num_workers=8, joint_jump_threshold=JOINT_JUMP_THRESHOLD, check_ee=True):
    npz_dir = os.path.abspath(npz_dir)
    if not os.path.isdir(npz_dir):
        raise NotADirectoryError(f"NPZ directory not found: {npz_dir}")

    if usd_dir is None:
        usd_dir = npz_dir
        print(f"[INFO] USD directory: same as NPZ ({npz_dir})")
    else:
        usd_dir = os.path.abspath(usd_dir)
        if not os.path.isdir(usd_dir):
            raise NotADirectoryError(f"USD directory not found: {usd_dir}")
        print(f"[INFO] USD directory: {usd_dir}")

    parent_dir = os.path.dirname(npz_dir)
    base_name = os.path.basename(npz_dir)
    unreachable_dir = os.path.join(parent_dir, f"{base_name}_unreachable")
    unreachable_usd_dir = os.path.join(parent_dir, f"{base_name}_unreachable_usd") if usd_dir != npz_dir else unreachable_dir

    os.makedirs(unreachable_dir, exist_ok=True)
    if usd_dir != npz_dir:
        os.makedirs(unreachable_usd_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    print(f"[INFO] Found {len(npz_files)} .npz files")

    if not npz_files:
        print("[WARN] No .npz files found - nothing to process")
        return

    chunks = [[] for _ in range(num_workers)]
    for i, fname in enumerate(npz_files):
        chunks[i % num_workers].append(fname)

    tasks = [
        (chunk, npz_dir, joint_jump_threshold, check_ee, i)
        for i, chunk in enumerate(chunks) if chunk
    ]

    print(f"[INFO] Starting filter ({len(tasks)} workers) | joint_jump={joint_jump_threshold:.4f} rad | EE check={'ON' if check_ee else 'OFF'}")
    start_time = time.time()

    all_results = []
    with mp.Pool(processes=len(tasks)) as pool:
        with tqdm(total=len(npz_files), desc="Filtering trajectories", unit="file") as pbar:
            for results in pool.imap_unordered(worker_process, tasks):
                all_results.extend(results)
                pbar.update(len(results))

    elapsed = time.time() - start_time

    print(f"[INFO] Moving bad files to output directories...")
    moved_count = 0
    moved_usd_count = 0
    jump_only = 0
    ee_only = 0
    both_issues = 0
    usd_missing = 0

    joint_labels = ("Joint jump", "Joint mutation (first 5 frames)")

    def _has_joint_issue(reasons):
        return any(r in reasons for r in joint_labels)

    for r in all_results:
        if not r["needs_move"]:
            continue

        fname = r["fname"]
        npz_src = os.path.join(npz_dir, fname)
        npz_dst = os.path.join(unreachable_dir, fname)

        try:
            shutil.move(npz_src, npz_dst)
            moved_count += 1

            has_j = _has_joint_issue(r["reasons"])
            has_ee = "EE unreachable" in r["reasons"]
            if has_j and has_ee:
                both_issues += 1
            elif has_j:
                jump_only += 1
            else:
                ee_only += 1

            usd_fname = os.path.splitext(fname)[0] + ".usd"
            usd_src = os.path.join(usd_dir, usd_fname)
            if os.path.exists(usd_src):
                usd_dst_dir = unreachable_usd_dir if usd_dir != npz_dir else unreachable_dir
                usd_dst = os.path.join(usd_dst_dir, usd_fname)
                try:
                    shutil.move(usd_src, usd_dst)
                    moved_usd_count += 1
                except Exception as e:
                    print(f"  Warning: USD move failed for {usd_fname}: {e}")
            else:
                usd_missing += 1

        except Exception as e:
            print(f"  Warning: NPZ move failed for {fname}: {e}")
            continue

    log_file = os.path.join(unreachable_dir, "unreachable_reasons.log")
    summary_file = os.path.join(unreachable_dir, "summary.txt")

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Filtering completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"Total files processed: {len(npz_files)}\n")
        lf.write(f"Bad files found: {moved_count}\n")
        lf.write(f"Joint jump threshold: {joint_jump_threshold:.4f} rad ({np.degrees(joint_jump_threshold):.2f}°)\n")
        lf.write(f"EE check: {'ENABLED' if check_ee else 'DISABLED'}\n")
        lf.write("=" * 70 + "\n\n")

        for r in sorted(all_results, key=lambda x: x["fname"]):
            if not r["needs_move"]:
                continue
            reason_str = " + ".join(r["reasons"])
            lf.write(f"{r['fname']}: {reason_str}\n")
            if r.get("error"):
                lf.write(f"  ERROR: {r['error']}\n")
            lf.write("\n")

    with open(summary_file, "w", encoding="utf-8") as sf:
        sf.write("=== TRAJECTORY FILTERING SUMMARY ===\n")
        sf.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sf.write(f"Processing time: {elapsed:.2f} seconds ({len(npz_files)/elapsed:.1f} files/sec)\n")
        sf.write(f"\nINPUT:\n")
        sf.write(f"  NPZ directory: {npz_dir}\n")
        sf.write(f"  USD directory: {usd_dir}\n")
        sf.write(f"\nOUTPUT:\n")
        sf.write(f"  Bad NPZ directory: {unreachable_dir}\n")
        if usd_dir != npz_dir:
            sf.write(f"  Bad USD directory: {unreachable_usd_dir}\n")
        sf.write(f"\nSTATISTICS:\n")
        sf.write(f"  Total files processed: {len(npz_files)}\n")
        sf.write(f"  Bad files moved: {moved_count} ({moved_count/len(npz_files)*100:.1f}%)\n")
        sf.write(f"    - Joint issue only (jump / first-5 mutation): {jump_only}\n")
        sf.write(f"    - EE unreachable only: {ee_only}\n")
        sf.write(f"    - Both joint + EE: {both_issues}\n")
        sf.write(f"  USD files moved: {moved_usd_count}\n")
        sf.write(f"  USD files missing: {usd_missing}\n")
        sf.write(f"\nCONFIGURATION:\n")
        sf.write(f"  Workers: {num_workers}\n")
        sf.write(f"  Joint jump threshold: {joint_jump_threshold:.4f} rad ({np.degrees(joint_jump_threshold):.2f}°)\n")
        sf.write(f"  EE unreachable check: {'ENABLED' if check_ee else 'DISABLED'}\n")
        sf.write(f"  Device: CPU (all workers)\n")

    print(f"\n{'=' * 60}")
    print(f"FILTERING COMPLETE ({elapsed:.2f}s | {len(npz_files)/elapsed:.1f} files/sec)")
    print(f"{'=' * 60}")
    print(f"Bad files moved: {moved_count}/{len(npz_files)} ({moved_count/len(npz_files)*100:.1f}%)")
    print(f"  ├── Joint issue only: {jump_only}")
    print(f"  ├── EE unreachable only: {ee_only}")
    print(f"  └── Joint + EE: {both_issues}")
    print(f"\nUSD handling:")
    print(f"  Moved successfully: {moved_usd_count}")
    print(f"  Missing (no .usd): {usd_missing}")
    print(f"\nOutput directories:")
    print(f"  NPZ: {unreachable_dir}")
    if usd_dir != npz_dir:
        print(f"  USD: {unreachable_usd_dir}")
    print(f"\nLogs:")
    print(f"  Detailed: {log_file}")
    print(f"  Summary:  {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter NPZ trajectories: joint jumps / first-5 mutations (always), optional EE reachability."
    )
    parser.add_argument("npz_dir", type=str, help="Directory containing .npz files")
    parser.add_argument("--usd-dir", type=str, default=None, help="Directory containing .usd files (default: same as npz_dir)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes (default: 8)")
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=JOINT_JUMP_THRESHOLD,
        help=f"Max allowed |Δq| per frame (radians, default {JOINT_JUMP_THRESHOLD})",
    )
    parser.add_argument(
        "--no-check-ee",
        action="store_true",
        help="Skip FK / EE unreachable check (joint discontinuity checks still run).",
    )
    args = parser.parse_args()

    max_workers = os.cpu_count()
    if args.workers > max_workers:
        print(f"[INFO] Limiting workers to {max_workers} (available CPU cores)")
        args.workers = max_workers

    main(
        npz_dir=args.npz_dir,
        usd_dir=args.usd_dir,
        num_workers=args.workers,
        joint_jump_threshold=args.jump_threshold,
        check_ee=not args.no_check_ee,
    )
