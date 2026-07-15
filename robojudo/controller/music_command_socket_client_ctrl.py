from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import MusicCommandSocketClientCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class MusicCommandSocketClientCtrl(Controller):
    """Connect to a remote TCP command server and expose received lines as commands."""

    cfg_ctrl: MusicCommandSocketClientCtrlCfg

    def __init__(self, cfg_ctrl: MusicCommandSocketClientCtrlCfg, env=None, device: str = "cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self._host = str(cfg_ctrl.host)
        self._port = int(cfg_ctrl.port)
        self._topic = str(cfg_ctrl.subscribe_topic)
        self._timeout_s = max(0.1, float(cfg_ctrl.connect_timeout_s))
        self._reconnect_interval_s = max(0.1, float(cfg_ctrl.reconnect_interval_s))
        self._queue: deque[str] = deque(maxlen=max(1, int(cfg_ctrl.queue_maxlen)))
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._client_loop,
            name="MusicCommandSocketClientCtrl",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[MusicCommandSocketClientCtrl] client thread started for %s:%d topic=%s",
            self._host,
            self._port,
            self._topic,
        )

    def reset(self):
        with self._lock:
            self._queue.clear()

    def shutdown(self) -> None:
        self._stop.set()

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

    def _enqueue(self, cmd: str) -> None:
        if not cmd:
            return
        with self._lock:
            self._queue.append(cmd)

    def _consume_buffer(self, buf: bytes) -> bytes:
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text or text.upper() == "ACK":
                continue
            self._enqueue(text)
            logger.info("[MusicCommandSocketClientCtrl] <- %s", text)
        return buf

    def _client_loop(self) -> None:
        while not self._stop.is_set():
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
                sock.settimeout(self._timeout_s)
                logger.info(
                    "[MusicCommandSocketClientCtrl] connected to %s:%d",
                    self._host,
                    self._port,
                )

                subscribe = f"SUBSCRIBE {self._topic}\n".encode("utf-8")
                sock.sendall(subscribe)
                buf = b""
                try:
                    first = sock.recv(4096)
                    if first:
                        buf += first
                        buf = self._consume_buffer(buf)
                except OSError:
                    pass

                while not self._stop.is_set():
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("remote closed connection")
                    buf += chunk
                    buf = self._consume_buffer(buf)
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    logger.warning(
                        "[MusicCommandSocketClientCtrl] connect/recv failed %s:%d: %s",
                        self._host,
                        self._port,
                        exc,
                    )
                    time.sleep(self._reconnect_interval_s)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
