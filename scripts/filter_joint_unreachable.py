#!/usr/bin/env python3
"""
Multi-PROCESS filter for .npz trajectories with unreachable end-effector positions.

Only needed when replay used rigid-transform augmentation (cloth position randomization).
Moves bad .npz AND corresponding .usd files to <input_dir>_unreachable/.

Uses process isolation (spawn) to avoid Warp/Newton global state conflicts.
"""

import os
import sys
import shutil
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp


mp.set_start_method('spawn', force=True)


left_ee_body_names = {"left_link16"}
right_ee_body_names = {"right_link26"}


def init_robot_fk(device="cpu"):
    """
    Initialize RobotFK in isolated process context.
    MUST be called inside each worker process AFTER fork/spawn.
    """
    import warp as wp
    import newton
    import newton.examples

    class RobotFK:
        def __init__(self, device="cpu"):
            self.device = device
            self._build_model()
            self._identify_ee_indices()
            self._init_state()

        def _build_model(self):
            builder = newton.ModelBuilder()
            urdf_path = newton.examples.get_asset("lift2_collision/lift2_collision.urdf")
            xform_base = wp.transform(p=wp.vec3(0.0, 0.0, 0.17))
            builder.add_urdf(
                urdf_path,
                floating=False,
                enable_self_collisions=False,
                xform=xform_base
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


def worker_process(task_args):
    """
    Single-process worker: NO shared state with other processes.
    Returns list of results for its file chunk.
    """
    (file_chunk, npz_dir, worker_id) = task_args

    fk = init_robot_fk(device="cpu")

    results = []
    for fname in file_chunk:
        npz_path = os.path.join(npz_dir, fname)
        result = {
            'fname': fname,
            'reasons': [],
            'needs_move': False,
            'error': None
        }

        try:
            data = np.load(npz_path)
            if 'joint_q' not in data:
                continue
            joint_q_seq = data['joint_q']
            if joint_q_seq.ndim != 2 or joint_q_seq.shape[1] != fk.model.joint_dof_count:
                continue

            left_pos, right_pos = fk.joint_to_ee_pos(joint_q_seq)
            init_left_x = left_pos[0, 0]
            init_right_x = right_pos[0, 0]
            if np.any(left_pos[:, 0] < init_left_x) or np.any(right_pos[:, 0] < init_right_x):
                result['reasons'].append("EE unreachable")

            result['needs_move'] = bool(result['reasons'])
            results.append(result)

        except Exception as e:
            result['error'] = str(e)
            results.append(result)
            continue

    return results


def main(npz_dir, usd_dir=None, num_workers=8):
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

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f"[INFO] Found {len(npz_files)} .npz files")

    if not npz_files:
        print("[WARN] No .npz files found - nothing to process")
        return

    chunks = [[] for _ in range(num_workers)]
    for i, fname in enumerate(npz_files):
        chunks[i % num_workers].append(fname)

    tasks = [
        (chunk, npz_dir, i)
        for i, chunk in enumerate(chunks) if chunk
    ]

    print(f"[INFO] Starting EE reachability filter with {len(tasks)} workers...")
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
    usd_missing = 0

    for r in all_results:
        if not r['needs_move']:
            continue

        fname = r['fname']
        npz_src = os.path.join(npz_dir, fname)
        npz_dst = os.path.join(unreachable_dir, fname)

        try:
            shutil.move(npz_src, npz_dst)
            moved_count += 1

            usd_fname = os.path.splitext(fname)[0] + '.usd'
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

    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(f"Filtering completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"Total files processed: {len(npz_files)}\n")
        lf.write(f"Bad files found: {moved_count}\n")
        lf.write("=" * 70 + "\n\n")

        for r in sorted(all_results, key=lambda x: x['fname']):
            if not r['needs_move']:
                continue
            reason_str = " + ".join(r['reasons'])
            lf.write(f"{r['fname']}: {reason_str}\n")
            if r.get('error'):
                lf.write(f"  ERROR: {r['error']}\n")
            lf.write("\n")

    with open(summary_file, 'w', encoding='utf-8') as sf:
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
        sf.write(f"  EE unreachable moved: {moved_count} ({moved_count/len(npz_files)*100:.1f}%)\n")
        sf.write(f"  USD files moved: {moved_usd_count}\n")
        sf.write(f"  USD files missing: {usd_missing}\n")
        sf.write(f"\nCONFIGURATION:\n")
        sf.write(f"  Workers: {num_workers}\n")
        sf.write(f"  Device: CPU (all workers)\n")

    print(f"\n{'=' * 60}")
    print(f"FILTERING COMPLETE ({elapsed:.2f}s | {len(npz_files)/elapsed:.1f} files/sec)")
    print(f"{'=' * 60}")
    print(f"EE unreachable moved: {moved_count}/{len(npz_files)} ({moved_count/len(npz_files)*100:.1f}%)")
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
        description="Filter trajectories with unreachable EE positions (only needed for rigid-transform augmented data)."
    )
    parser.add_argument("npz_dir", type=str, help="Directory containing .npz files")
    parser.add_argument("--usd-dir", type=str, default=None,
                        help="Directory containing .usd files (default: same as npz_dir)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes (default: 8)")
    args = parser.parse_args()

    max_workers = os.cpu_count()
    if args.workers > max_workers:
        print(f"[INFO] Limiting workers to {max_workers} (available CPU cores)")
        args.workers = max_workers

    main(
        npz_dir=args.npz_dir,
        usd_dir=args.usd_dir,
        num_workers=args.workers,
    )
