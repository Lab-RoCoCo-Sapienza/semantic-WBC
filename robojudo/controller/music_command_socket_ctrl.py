from __future__ import annotations

import json
import logging
import socket
import threading
from collections import deque

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import MusicCommandSocketCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class MusicCommandSocketCtrl(Controller):
    """Receive remote command strings over TCP and expose them as pipeline commands."""

    cfg_ctrl: MusicCommandSocketCtrlCfg

    def __init__(self, cfg_ctrl: MusicCommandSocketCtrlCfg, env=None, device: str = "cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._host = cfg_ctrl.host
        self._port = int(cfg_ctrl.port)
        self._queue: deque[str] = deque(maxlen=max(1, int(cfg_ctrl.queue_maxlen)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

        self._thread = threading.Thread(
            target=self._server_loop,
            name="MusicCommandSocketCtrl",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[MusicCommandSocketCtrl] server thread started for %s:%d (bind is async; CRITICAL on failure)",
            self._host,
            self._port,
        )

    def reset(self):
        with self._lock:
            self._queue.clear()

    def shutdown(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def get_data(self):
        with self._lock:
            n = len(self._queue)
        return {"pending_remote_commands": n}

    def process_triggers(self, ctrl_data):
        commands: list[str] = []
        with self._lock:
            while self._queue:
                commands.append(self._queue.popleft())
        return ctrl_data, commands

    def _server_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._host, self._port))
        except OSError as exc:
            logger.critical(
                "[MusicCommandSocketCtrl] cannot bind %s:%d: %s. "
                "Remote music commands will NOT be received; free the port or set ROBOJUDO_MUSIC_PORT.",
                self._host,
                self._port,
                exc,
            )
            return
        sock.listen(4)
        logger.info("[MusicCommandSocketCtrl] listening on %s:%d", self._host, self._port)
        sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue
            with conn:
                conn.settimeout(2.0)
                peer = f"{addr[0]}:{addr[1]}"
                try:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            raw, buf = buf.split(b"\n", 1)
                            response = self._consume_line(raw.decode("utf-8", errors="replace"), peer)
                            if response:
                                try:
                                    conn.sendall((response + "\n").encode("utf-8"))
                                except OSError:
                                    break
                except OSError:
                    continue

    def _consume_line(self, line: str, peer: str) -> str | None:
        text = line.strip()
        if not text:
            return None
        if text.upper().startswith("SUBSCRIBE"):
            # Compatibility handshake: client may only send "SUBSCRIBE <topic>" and
            # optionally wait for one line of ack.
            logger.info("[MusicCommandSocketCtrl] %s -> %s (handshake)", peer, text)
            return "ACK"
        cmd: str | None = None
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    v = payload.get("command")
                    if isinstance(v, str):
                        cmd = v.strip()
            except json.JSONDecodeError:
                cmd = None
        else:
            cmd = text
        if not cmd:
            return None
        with self._lock:
            self._queue.append(cmd)
        logger.info("[MusicCommandSocketCtrl] %s -> %s", peer, cmd)
        return None
