import argparse
import os
import sys

# Project root must be on sys.path when running: python apps/teleoperation_app.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs import load_task_from_name


def run_task(task):
    print("\n=== Keyboard Teleoperation Controls ===")
    print("Left gripper:  W/S=front/back, A/D=left/right, Q/E=down/up, X=toggle grip")
    print("Right gripper: I/K=front/back, J/L=left/right, U/O=down/up, M=toggle grip")
    print("Camera: Arrow keys=move, Mouse left-drag=look, Scroll=zoom")
    print("Other:  H=toggle UI, Space=pause, 3/4/5=camera presets, ESC=quit, R=full reset")
    print("========================================\n")

    if not hasattr(task.recorder, "save_image"):
        task.recorder.save_image = lambda viewer, frame: None

    while task.viewer.is_running():
        if hasattr(task, "viewer") and hasattr(task.viewer, "is_key_down"):
            if task.viewer.is_key_down("r"):
                python = sys.executable
                os.execv(python, [python] + sys.argv)

        if not task.viewer.is_paused():
            task.step()
        task.render()

    task.finished()
    task.viewer.close()


def run_task_stream(task, host, ws_port, http_port):
    """Run task with WebSocket streaming for remote teleoperation."""
    from stream import StreamServer

    server = StreamServer(host=host, ws_port=ws_port, http_port=http_port)
    server.start()
    server.hook_viewer(task.viewer)

    print("\n=== Stream Teleoperation Mode ===")
    print(f"WebSocket: ws://{host}:{ws_port}")
    print(f"Web UI:    http://{host}:{http_port}")
    print("Local keyboard remains active; WebSocket keys are merged with local input.")
    print("Use Reset on the web page or send a reset message to reset the simulation.")
    print("===================================\n")

    if not hasattr(task.recorder, "save_image"):
        task.recorder.save_image = lambda viewer, frame: None

    def _full_restart():
        print("[Stream] Restarting simulation process...")
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    while task.viewer.is_running():
        if server.key_state.consume_reset():
            _full_restart()

        if hasattr(task, "viewer") and hasattr(task.viewer, "is_key_down"):
            if task.viewer.is_key_down("r"):
                _full_restart()

        if not task.viewer.is_paused():
            task.step()
        task.render()

        server.capture_frame(task.viewer)

    task.finished()
    task.viewer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-arm cloth teleoperation")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable WebSocket stream mode (remote keys + video frames)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Stream mode bind address (default 0.0.0.0 for all interfaces)",
    )
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default 8765)")
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP static file server port (default 8080)",
    )
    parser.add_argument("--task", default="lift_manip_shirt", help="Task name (default lift_manip_shirt)")
    return parser.parse_args()


def main():
    args = parse_args()
    task = load_task_from_name(args.task)

    if args.stream:
        run_task_stream(task, args.host, args.ws_port, args.http_port)
    else:
        run_task(task)


if __name__ == "__main__":
    main()
