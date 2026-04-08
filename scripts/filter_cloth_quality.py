#!/usr/bin/env python3
"""
Rule-based cloth quality filter for replay output (npz + usd pairs).

Reads each USD's last-frame cloth mesh vertices at /root/model/triangles,
checks z_max / y_min thresholds, and moves bad npz+usd pairs to a
separate directory.

Two modes:
  1. Direct mode (no rigid transform): apply thresholds directly on raw vertices.
  2. Aligned mode (--ref-usd): when replay used cloth position randomization,
     first undo the per-USD rigid transform via Kabsch alignment, then filter
     on aligned vertices.

Typical workflow:
  A) No rigid transform:
       replay -> filter_cloth_quality.py -> render

  B) With rigid transform:
       replay -> filter_joint_unreachable.py (EE filter)
              -> filter_cloth_quality.py --ref-usd <ref.usd> (cloth quality)
              -> render
"""

import argparse
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from pxr import Usd, UsdGeom
except ImportError:
    raise ImportError(
        "USD library (pxr) is required. Install via: pip install usd-core"
    )


PRIM_PATH = "/root/model/triangles"

ZMAX_LO = 0.935
ZMAX_HI = 0.960
YMIN_THRESHOLD = -0.090
DEFAULT_MAX_KABSCH_RMSE = 0.05  # metres


def list_usd_files(folder: str) -> List[str]:
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith((".usd", ".usda", ".usdc"))
    )


def load_verts(stage, frame: float) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(PRIM_PATH)
    if not prim.IsValid():
        return None
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(time=frame)
    if pts is None:
        return None
    v = np.asarray(pts, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        return None
    return v


def kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute R, t such that dst ~ R @ src + t. Returns (R, t, rmse)."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - R @ src_c
    rmse = float(np.sqrt(((dst - (R @ src.T).T - t) ** 2).sum(axis=1).mean()))
    return R, t, rmse


def passes_filter(z_max: float, y_min: float,
                  zmax_lo: float, zmax_hi: float, ymin_thresh: float) -> bool:
    return zmax_lo <= z_max <= zmax_hi and y_min > ymin_thresh


def fail_reasons(z_max: float, y_min: float,
                 zmax_lo: float, zmax_hi: float, ymin_thresh: float) -> str:
    reasons = []
    if z_max < zmax_lo:
        reasons.append(f"z_max={z_max:.4f}<{zmax_lo}")
    elif z_max > zmax_hi:
        reasons.append(f"z_max={z_max:.4f}>{zmax_hi}")
    if y_min <= ymin_thresh:
        reasons.append(f"y_min={y_min:.4f}<={ymin_thresh}")
    return " & ".join(reasons)


def process_direct(usd_path: str, zmax_lo, zmax_hi, ymin_thresh):
    """Direct mode: read last-frame vertices and check thresholds."""
    try:
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            return None
        end = stage.GetEndTimeCode()
        verts = load_verts(stage, end)
        if verts is None:
            return None
        z_max = float(verts[:, 2].max())
        y_min = float(verts[:, 1].min())
        passed = passes_filter(z_max, y_min, zmax_lo, zmax_hi, ymin_thresh)
        return {
            "z_max": z_max, "y_min": y_min,
            "passed": passed,
            "reason": "" if passed else fail_reasons(z_max, y_min, zmax_lo, zmax_hi, ymin_thresh),
        }
    except Exception as e:
        return {"error": str(e)}


def process_aligned(usd_path: str, ref_f0: np.ndarray, max_rmse: float,
                    zmax_lo, zmax_hi, ymin_thresh):
    """Aligned mode: Kabsch undo rigid transform, then check thresholds."""
    try:
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            return None
        end = stage.GetEndTimeCode()
        dst_f0 = load_verts(stage, 0)
        dst_last = load_verts(stage, end)
        if dst_f0 is None or dst_last is None:
            return None
        if dst_f0.shape[0] != ref_f0.shape[0]:
            return {"error": f"vertex count mismatch: {dst_f0.shape[0]} vs ref {ref_f0.shape[0]}"}

        R, t, rmse = kabsch(ref_f0, dst_f0)
        if rmse > max_rmse:
            return {"error": f"kabsch_rmse={rmse*1000:.1f}mm > {max_rmse*1000:.0f}mm"}

        aligned = ((dst_last - t) @ R).astype(np.float32)
        z_max = float(aligned[:, 2].max())
        y_min = float(aligned[:, 1].min())
        passed = passes_filter(z_max, y_min, zmax_lo, zmax_hi, ymin_thresh)
        return {
            "z_max": z_max, "y_min": y_min,
            "kabsch_rmse_mm": rmse * 1000,
            "passed": passed,
            "reason": "" if passed else fail_reasons(z_max, y_min, zmax_lo, zmax_hi, ymin_thresh),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based cloth quality filter on replay USD+NPZ pairs.",
        epilog=(
            "Without --ref-usd: direct threshold on raw vertices.\n"
            "With    --ref-usd: Kabsch-align to reference first (for rigid-transform data)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session_dir",
        help="Replay session directory (contains npz/ and usd/ sub-folders), "
             "e.g. ./replay/my_replay_0001",
    )
    parser.add_argument(
        "--ref-usd", default=None,
        help="Reference USD for Kabsch alignment (required for rigid-transform augmented data). "
             "Use any 'known-good' USD with unmodified cloth position at frame 0.",
    )
    parser.add_argument("--zmax-lo", type=float, default=ZMAX_LO)
    parser.add_argument("--zmax-hi", type=float, default=ZMAX_HI)
    parser.add_argument("--ymin-thresh", type=float, default=YMIN_THRESHOLD)
    parser.add_argument("--max-kabsch-rmse", type=float, default=DEFAULT_MAX_KABSCH_RMSE,
                        help="Max Kabsch RMSE at frame 0 (metres); only used with --ref-usd")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without moving files")
    args = parser.parse_args()

    usd_dir = os.path.join(args.session_dir, "usd")
    npz_dir = os.path.join(args.session_dir, "npz")

    if not os.path.isdir(usd_dir):
        raise FileNotFoundError(f"USD directory not found: {usd_dir}")
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    aligned_mode = args.ref_usd is not None
    ref_f0 = None
    if aligned_mode:
        ref_stage = Usd.Stage.Open(args.ref_usd)
        ref_f0 = load_verts(ref_stage, 0)
        if ref_f0 is None:
            raise RuntimeError(f"Cannot load frame-0 vertices from {args.ref_usd}")
        print(f"Mode       : aligned (Kabsch)")
        print(f"Reference  : {args.ref_usd}")
        print(f"  frame-0 vertices: {len(ref_f0)}")
        print(f"  max Kabsch RMSE : {args.max_kabsch_rmse*1000:.0f} mm")
    else:
        print(f"Mode       : direct (no rigid transform)")

    print(f"Thresholds : z_max in [{args.zmax_lo}, {args.zmax_hi}], y_min > {args.ymin_thresh}")
    print(f"Session    : {args.session_dir}")
    print()

    usd_files = list_usd_files(usd_dir)
    print(f"USD files  : {len(usd_files)}")

    bad_usd_dir = os.path.join(args.session_dir, "usd_bad_cloth")
    bad_npz_dir = os.path.join(args.session_dir, "npz_bad_cloth")

    passed_list = []
    failed_list = []
    skipped = 0

    for fname in tqdm(usd_files, desc="Filtering"):
        usd_path = os.path.join(usd_dir, fname)

        if aligned_mode:
            result = process_aligned(
                usd_path, ref_f0, args.max_kabsch_rmse,
                args.zmax_lo, args.zmax_hi, args.ymin_thresh,
            )
        else:
            result = process_direct(
                usd_path, args.zmax_lo, args.zmax_hi, args.ymin_thresh,
            )

        if result is None or "error" in result:
            skipped += 1
            if result and "error" in result:
                tqdm.write(f"  [skip] {fname}: {result['error']}")
            continue

        result["file"] = fname
        if result["passed"]:
            passed_list.append(result)
        else:
            failed_list.append(result)

    print(f"\n{'=' * 60}")
    print(f"Passed : {len(passed_list)} / {len(usd_files)}  (skipped {skipped})")
    print(f"Failed : {len(failed_list)}")

    if passed_list:
        z_maxs = np.array([r["z_max"] for r in passed_list])
        y_mins = np.array([r["y_min"] for r in passed_list])
        print(f"  z_max: mean={z_maxs.mean():.4f}, min={z_maxs.min():.4f}, max={z_maxs.max():.4f}")
        print(f"  y_min: mean={y_mins.mean():.4f}, min={y_mins.min():.4f}, max={y_mins.max():.4f}")

    if args.dry_run:
        print("\n(dry run -- no files moved)")
        return

    # Move bad files
    if not failed_list:
        print("\nNo bad files to move.")
    else:
        os.makedirs(bad_usd_dir, exist_ok=True)
        os.makedirs(bad_npz_dir, exist_ok=True)
        moved_usd = 0
        moved_npz = 0

        for r in failed_list:
            fname = r["file"]

            usd_src = os.path.join(usd_dir, fname)
            if os.path.exists(usd_src):
                shutil.move(usd_src, os.path.join(bad_usd_dir, fname))
                moved_usd += 1

            npz_name = os.path.splitext(fname)[0] + ".npz"
            npz_src = os.path.join(npz_dir, npz_name)
            if os.path.exists(npz_src):
                shutil.move(npz_src, os.path.join(bad_npz_dir, npz_name))
                moved_npz += 1

        print(f"\nMoved {moved_usd} bad USD  -> {bad_usd_dir}")
        print(f"Moved {moved_npz} bad NPZ  -> {bad_npz_dir}")

    # Summary log
    summary_path = os.path.join(args.session_dir, "cloth_filter_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Cloth quality filter summary\n")
        f.write(f"Mode: {'aligned' if aligned_mode else 'direct'}\n")
        if aligned_mode:
            f.write(f"Reference: {args.ref_usd}\n")
            f.write(f"Max Kabsch RMSE: {args.max_kabsch_rmse*1000:.0f} mm\n")
        f.write(f"Thresholds: z_max in [{args.zmax_lo}, {args.zmax_hi}], y_min > {args.ymin_thresh}\n")
        f.write(f"Session: {args.session_dir}\n\n")
        f.write(f"Passed: {len(passed_list)}\nFailed: {len(failed_list)}\nSkipped: {skipped}\n\n")

        f.write("=== PASSED ===\n")
        for r in passed_list:
            line = f"  {r['file']}  z_max={r['z_max']:.6f}  y_min={r['y_min']:.6f}"
            if "kabsch_rmse_mm" in r:
                line += f"  kabsch={r['kabsch_rmse_mm']:.2f}mm"
            f.write(line + "\n")

        f.write("\n=== FAILED ===\n")
        for r in failed_list:
            line = f"  {r['file']}  z_max={r['z_max']:.6f}  y_min={r['y_min']:.6f}"
            if "kabsch_rmse_mm" in r:
                line += f"  kabsch={r['kabsch_rmse_mm']:.2f}mm"
            line += f"  reason: {r['reason']}"
            f.write(line + "\n")

    print(f"\nSummary: {summary_path}")
    remaining_usd = len([f for f in os.listdir(usd_dir) if f.endswith((".usd", ".usda", ".usdc"))])
    remaining_npz = len([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    print(f"\nRemaining for rendering: {remaining_usd} USD, {remaining_npz} NPZ")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
