"""
SIM1-Sim Replay: Replay recorded trajectories as USD/NPZ output.
"""

import argparse
import os
import re
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import newton
import warp as wp

from components.randomization import Randomizer
from components.recorder import DualArmRecorder
from envs import LiftShortShirt
from tasks import AconeClothManip


def get_project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DummyViewer:
    """Headless viewer shim (no OpenGL/GUI/VNC required)."""

    def __init__(self, fps=60):
        self.sim_time = 0.0
        self.fps = fps

    def begin_frame(self, time):
        pass

    def end_frame(self):
        pass

    def log_state(self, state):
        pass

    def log_gizmo(self, name, tf):
        pass

    def log_contacts(self, contacts, state):
        pass

    def set_model(self, model):
        pass

    def is_running(self):
        return True

    def is_paused(self):
        return False

    def is_key_down(self, key):
        return False

    def close(self):
        pass


def run_replay(task):
    try:
        frame = 0
        # With ViewerGL, also stop when the user closes the window.
        while not task.is_finished and task.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                task.step()
            with wp.ScopedTimer("render", active=False):
                task.render()
            frame += 1
        if task.is_finished:
            print(f"[INFO] Simulation finished after {frame} frames.")
        else:
            print(f"[INFO] Stopped after {frame} frames (viewer closed or interrupted).")
    finally:
        task.viewer.close()
        task.finished()


def parse_args():
    parser = argparse.ArgumentParser(description="SIM1-Sim Replay: replay .npz trajectory(ies)")
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to a single .npz file, or a directory whose *.npz files are replayed in sorted order",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="replay_output",
        help=(
            "Base name for a numbered session under replay/: creates replay/<name>_NNNN/ "
            "with npz/ and usd/ (and video/ if --save_video). NNNN auto-increments per new session."
        ),
    )
    parser.add_argument(
        "--save_usd",
        action="store_true",
        default=True,
        help="Save USD output (default: True)",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Save video from OpenGL frames (requires --no-headless; default: False)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without an OpenGL window (default: --headless). Use --no-headless to show ViewerGL.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Simulation FPS (default: 60)")
    parser.add_argument(
        "--sim_substeps", type=int, default=24, help="Simulation substeps per frame (default: 24)"
    )
    parser.add_argument(
        "--position-randomize",
        action="store_true",
        default=False,
        help=(
            "Randomize cloth pose at replay (±2 cm XY translation, ±15° yaw). "
            "Default: off (fixed cloth pose)."
        ),
    )
    args = parser.parse_args()
    if args.save_video and args.headless:
        parser.error("--save-video needs an OpenGL viewer; run with --no-headless")
    return args


def make_viewer(headless: bool, fps: int):
    if headless:
        return DummyViewer(fps=fps)
    return newton.viewer.ViewerGL(headless=False)


def resolve_npz_path(path: str) -> str:
    root = get_project_root()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return os.path.normpath(path)


def allocate_replay_session_dir(replay_root: str, base_name: str) -> str:
    """
    Create replay/<base_name>_NNNN/ with npz/, usd/, and video/ subdirs.
    NNNN is the next unused 4-digit suffix among existing directories under replay_root.
    """
    os.makedirs(replay_root, exist_ok=True)
    pat = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})$")
    max_n = 0
    for name in os.listdir(replay_root):
        if not os.path.isdir(os.path.join(replay_root, name)):
            continue
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    next_n = max_n + 1
    session_name = f"{base_name}_{next_n:04d}"
    session_path = os.path.join(replay_root, session_name)
    os.makedirs(os.path.join(session_path, "npz"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "usd"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "video"), exist_ok=True)
    return session_path


def load_cloth_pos_rot_from_npz(npz_file_path: str):
    """
    If the NPZ stores `pos_rot` with shape (6,), use it as cloth world pose at replay
    (x, y, z, euler_x, euler_y, euler_z in radians). Otherwise use LiftShortShirt.CLOTH_POS.
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        if "pos_rot" not in data:
            return None
        pr = np.asarray(data["pos_rot"], dtype=np.float64).reshape(-1)
        if pr.size != 6:
            print(
                f"[WARN] pos_rot has size {pr.size}, expected 6; "
                "using default CLOTH_POS from LiftShortShirt."
            )
            return None
        return pr[:6].copy()
    except Exception as e:
        print(f"[WARN] Could not read pos_rot from {npz_file_path}: {e}")
        return None


def collect_npz_files(path: str) -> list[str]:
    """Return a sorted list of .npz paths: one file, or all *.npz in a directory."""
    if not os.path.exists(path):
        print(f"Error: Path not found - {path}")
        sys.exit(1)
    if os.path.isfile(path):
        if not path.lower().endswith(".npz"):
            print(f"Error: Not a .npz file - {path}")
            sys.exit(1)
        return [path]
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path) if f.lower().endswith(".npz"))
        paths = [os.path.join(path, f) for f in names]
        if not paths:
            print(f"Error: No .npz files in directory - {path}")
            sys.exit(1)
        return paths
    print(f"Error: Invalid path - {path}")
    sys.exit(1)


def run_single_replay(npz_file_path: str, args, session_dir: str) -> None:
    viewer = make_viewer(args.headless, args.fps)

    if args.position_randomize:
        randomizer = Randomizer(
            pos_range=[[-0.02, 0.02], [-0.02, 0.02], [0.0, 0.0]],
            rot_range=[[0, 0], [0, 0], [-15, 15]],
            axis=[-1, 1, 1],
        )
    else:
        randomizer = Randomizer(
            pos_range=[[-0.0, 0.0], [-0.0, 0.0], [0.0, 0.0]],
            rot_range=[[0, 0], [0, 0], [0, 0]],
            axis=[-1, 1, 1],
        )

    cloth_pos_rot = load_cloth_pos_rot_from_npz(npz_file_path)
    env = LiftShortShirt(randomizer, cloth_pos_rot=cloth_pos_rot)
    recorder = DualArmRecorder("acone/test", env.robot_joint_q_cnt, mode="replay", fps=args.fps)
    recorder.load_from_file(npz_file_path)

    task = AconeClothManip(
        viewer=viewer,
        env=env,
        recorder=recorder,
        sim_substeps=args.sim_substeps,
        mode="replay",
        save_video=args.save_video,
        save_usd=args.save_usd,
        folder_name=args.folder_name,
        replay_session_dir=session_dir,
        input_npz_path=npz_file_path,
    )

    run_replay(task)


def main():
    args = parse_args()
    npz_path = resolve_npz_path(args.npz_path)
    npz_list = collect_npz_files(npz_path)

    replay_root = os.path.join(get_project_root(), "replay")
    session_dir = allocate_replay_session_dir(replay_root, args.folder_name)
    print(f"[INFO] Session output directory: {session_dir}")
    print(f"[INFO]   npz/  — augmented replay .npz (pairs with usd/ by basename)")
    print(f"[INFO]   usd/  — simulation USD for the same trajectories")

    total = len(npz_list)
    for i, npz_file_path in enumerate(npz_list, start=1):
        print(f"\n{'=' * 60}\n[INFO] Replay {i}/{total}: {npz_file_path}\n{'=' * 60}")
        run_single_replay(npz_file_path, args, session_dir)


if __name__ == "__main__":
    main()
