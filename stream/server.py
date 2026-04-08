"""
WebSocket streaming server for remote teleoperation.

Architecture:
- Only ONE active controller at a time; others wait in a FIFO queue
- Active controller idle > 2 min → auto-kick, next in queue promoted
- All connected clients (active + queued) receive video frames (spectate)
- Only the active controller's key input is forwarded to the simulation
"""

import asyncio
import threading
import json
import time
import functools
import http.server
import os

import cv2
import numpy as np


class KeyStateManager:
    """Thread-safe key state shared between WebSocket thread and simulation main loop."""

    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self._reset_requested = False
        self._last_client_ts = None

    def update(self, keys, client_ts=None):
        with self._lock:
            self._pressed = set(keys)
            if client_ts is not None:
                self._last_client_ts = client_ts

    def is_pressed(self, key):
        with self._lock:
            return key in self._pressed

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset(self):
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False

    def pop_client_ts(self):
        with self._lock:
            ts = self._last_client_ts
            self._last_client_ts = None
            return ts


class FrameBuffer:
    """Thread-safe single-frame buffer. The sim loop writes; WebSocket senders read."""

    def __init__(self):
        self._frame_bytes = None
        self._lock = threading.Lock()
        self._frame_id = 0

    def update(self, jpeg_bytes):
        with self._lock:
            self._frame_bytes = jpeg_bytes
            self._frame_id += 1

    def get(self):
        with self._lock:
            return self._frame_bytes, self._frame_id


class StreamServer:
    """
    Manages WebSocket + HTTP servers for remote teleoperation
    with single-active-controller queue management.
    """

    def __init__(self, host="0.0.0.0", ws_port=8765, http_port=8080,
                 idle_timeout=120):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.web_dir = os.path.join(os.path.dirname(__file__), "web")

        self.key_state = KeyStateManager()
        self.frame_buffer = FrameBuffer()

        self._ws_thread = None
        self._http_thread = None
        self._loop = None
        self._connections = set()

        self._jpeg_quality = 75
        self._target_send_fps = 30

        # Queue management (accessed only from asyncio thread, no lock needed)
        self._active_ws = None
        self._queue = []
        self._last_activity = None
        self._idle_timeout = idle_timeout

    def start(self):
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()

        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

        time.sleep(0.3)

    # ── WebSocket server ──────────────────────────────────────────

    def _run_ws(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve_ws())

    async def _serve_ws(self):
        import websockets

        async with websockets.serve(
            self._ws_handler,
            self.host,
            self.ws_port,
            max_size=2**20,
            ping_interval=20,
            ping_timeout=60,
        ):
            print(f"[Stream] WebSocket server: ws://{self.host}:{self.ws_port}")
            await asyncio.Future()

    async def _ws_handler(self, websocket):
        self._connections.add(websocket)
        addr = websocket.remote_address

        if self._active_ws is None:
            await self._set_active(websocket)
            print(f"[Queue] {addr} -> controller (queue 0)")
        else:
            self._queue.append(websocket)
            await self._broadcast_queue_status()
            print(f"[Queue] {addr} -> queued #{len(self._queue)} (queue size {len(self._queue)})")

        try:
            recv_task = asyncio.create_task(self._receive_from_client(websocket))
            send_task = asyncio.create_task(self._send_frames(websocket))
            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            self._connections.discard(websocket)
            if websocket is self._active_ws:
                # Active controller disconnected: request a reset so sim restarts on disconnect
                self.key_state.request_reset()
                self._active_ws = None
                self._last_activity = None
                self.key_state.update([])
                print(f"[Queue] Controller disconnected: {addr}")
                await self._promote_next()
            elif websocket in self._queue:
                self._queue.remove(websocket)
                await self._broadcast_queue_status()
                print(f"[Queue] Queued client left: {addr} (queue size {len(self._queue)})")

    # ── Queue management ──────────────────────────────────────────

    async def _set_active(self, websocket):
        self._active_ws = websocket
        self._last_activity = time.monotonic()
        try:
            await websocket.send(json.dumps({"type": "active"}))
        except Exception:
            pass

    async def _promote_next(self):
        while self._queue:
            next_ws = self._queue.pop(0)
            try:
                await self._set_active(next_ws)
                addr = next_ws.remote_address
                print(f"[Queue] Promoted: {addr} (queue size {len(self._queue)})")
                await self._broadcast_queue_status()
                return
            except Exception:
                self._connections.discard(next_ws)
                continue
        self.key_state.update([])
        print("[Queue] Queue empty, waiting for new connections")

    async def _broadcast_queue_status(self):
        for i, ws in enumerate(self._queue):
            try:
                await ws.send(json.dumps({
                    "type": "queue",
                    "position": i + 1,
                    "total": len(self._queue),
                }))
            except Exception:
                pass

    # ── Receive keys (only active controller's input matters) ─────

    async def _receive_from_client(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if websocket is not self._active_ws:
                    continue

                if msg_type == "keys":
                    keys = data.get("keys", [])
                    if keys:
                        self._last_activity = time.monotonic()
                    self.key_state.update(keys, client_ts=data.get("timestamp"))
                elif msg_type == "reset":
                    self._last_activity = time.monotonic()
                    self.key_state.request_reset()
            except (json.JSONDecodeError, Exception):
                pass

    # ── Send frames (to ALL clients) + idle kick for active ───────

    async def _send_frames(self, websocket):
        last_frame_id = -1
        interval = 1.0 / self._target_send_fps
        pong_interval = 0.5
        last_pong_time = 0.0

        while True:
            # Idle timeout: kick active controller if no key press for _idle_timeout
            if (websocket is self._active_ws
                    and self._last_activity is not None
                    and time.monotonic() - self._last_activity > self._idle_timeout):
                idle_sec = int(self._idle_timeout)
                print(f"[Queue] Controller {websocket.remote_address} idle {idle_sec}s, kicking")
                try:
                    await websocket.send(json.dumps({
                        "type": "kicked",
                        "reason": "idle",
                    }))
                    await websocket.close()
                except Exception:
                    pass
                # Request reset on idle kick so environment is clean for next user
                self.key_state.request_reset()
                break

            # Send video frame
            frame_bytes, frame_id = self.frame_buffer.get()
            if frame_bytes is not None and frame_id != last_frame_id:
                try:
                    await websocket.send(frame_bytes)
                    last_frame_id = frame_id
                except Exception:
                    break

            # Pong echo (only to active controller for RTT measurement)
            if websocket is self._active_ws:
                now = time.monotonic()
                if now - last_pong_time >= pong_interval:
                    client_ts = self.key_state.pop_client_ts()
                    if client_ts is not None:
                        try:
                            await websocket.send(json.dumps({
                                "type": "pong",
                                "client_ts": client_ts,
                            }))
                        except Exception:
                            break
                    last_pong_time = now

            await asyncio.sleep(interval)

    # ── HTTP static file server ───────────────────────────────────

    def _run_http(self):
        handler = functools.partial(
            _SilentHTTPHandler,
            directory=self.web_dir,
        )
        httpd = http.server.HTTPServer((self.host, self.http_port), handler)
        print(f"[Stream] HTTP server: http://{self.host}:{self.http_port}")
        print(f"[Stream] Open the URL in a browser to start remote teleoperation")
        httpd.serve_forever()

    # ── Frame capture (called from main sim loop) ─────────────────

    def capture_frame(self, viewer):
        if not self._connections:
            return
        try:
            frame_wp = viewer.get_frame()
            frame_np = frame_wp.numpy()
            bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            self.frame_buffer.update(buf.tobytes())
        except Exception:
            pass

    # ── Viewer hook ───────────────────────────────────────────────

    def hook_viewer(self, viewer):
        original_is_key_down = viewer.is_key_down
        key_state = self.key_state

        def patched_is_key_down(key):
            if key_state.is_pressed(key):
                return True
            return original_is_key_down(key)

        viewer.is_key_down = patched_is_key_down


class _SilentHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
