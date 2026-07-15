from __future__ import annotations

import contextlib
import http.server
import socket
import threading
import urllib.parse
import webbrowser


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/cmd":
            qs = urllib.parse.parse_qs(parsed.query)
            cmd = (qs.get("c", [""]) or [""])[0]
            if cmd:
                self.server.gui._emit(cmd)  # type: ignore[attr-defined]
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if parsed.path != "/":
            self.send_response(404)
            self.end_headers()
            return

        html = self.server.gui._render()  # type: ignore[attr-defined]
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        # Keep console clean
        return


class PolicyGUI:
    """
    Local web UI for sending RoboJuDo COMMANDS.
    Works on macOS without Tkinter/pygame main-thread issues.
    """

    def __init__(self, policy_ctrl, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
        self.policy_ctrl = policy_ctrl
        self.host = host
        self.port = port
        self.open_browser = open_browser

        self._server = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, command: str):
        self.policy_ctrl.gui_commands.put(command)

    def _render(self) -> str:
        def btn(label: str, cmd: str) -> str:
            q = urllib.parse.urlencode({"c": cmd})
            return f'<a class="btn" href="/cmd?{q}">{label}</a>'

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboJuDo Policy Panel</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; background:#0f1115; color:#eaeef5; }}
    .wrap {{ max-width: 520px; margin: 18px auto; padding: 0 12px; }}
    h2 {{ margin: 14px 0 10px; font-size: 16px; color:#cfd6e6; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .col {{ display:flex; flex-direction:column; gap:10px; }}
    .btn {{ display:block; padding: 12px 12px; border-radius: 10px; background:#1b1f2a; color:#eaeef5;
            text-decoration:none; border:1px solid #2b3242; }}
    .btn:hover {{ background:#232a38; }}
    .btn.small {{ padding: 10px 12px; }}
    .muted {{ color:#98a2b3; font-size: 12px; margin-top: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Unitree modes</h2>
    <div class="grid">
      {btn("W: Walk", "[POLICY_SWITCH],0")}
      {btn("S: Stand", "[POLICY_SWITCH],1")}
    </div>

    <h2>BeyondMimic models</h2>
    <div class="col">
      {btn("1: bts_2_0_tracking_2", "[POLICY_SWITCH],2")}
      {btn("2: bts_dynamite_tracking", "[POLICY_SWITCH],3")}
      {btn("3: easy_sample", "[POLICY_SWITCH],4")}
      {btn("4: Swim_tracking", "[POLICY_SWITCH],5")}
      {btn("5: thriller", "[POLICY_SWITCH],6")}
      {btn("6: salsa_tracking", "[POLICY_SWITCH],7")}
      {btn("7: gdance", "[POLICY_SWITCH],8")}
      {btn("8: Salsa_4", "[POLICY_SWITCH],9")}
      {btn("9: thriller_locked_waist", "[POLICY_SWITCH],10")}
    </div>

    <h2>System</h2>
    <div class="grid">
      {btn("R: Reborn", "[SIM_REBORN]")}
      {btn("Esc Shutdown", "[SHUTDOWN]")}
    </div>

    <div class="muted">Local UI: http://{self.host}:{self.port}</div>
  </div>
</body>
</html>
"""

    def _bind_port(self) -> int:
        # try preferred port, else pick an ephemeral one
        for p in (self.port, 0):
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((self.host, p))
                    return s.getsockname()[1]
                except OSError:
                    continue
        return self.port

    def _run(self):
        port = self._bind_port()
        self.port = port
        server = http.server.ThreadingHTTPServer((self.host, port), _Handler)
        server.gui = self  # type: ignore[attr-defined]
        self._server = server

        if self.open_browser:
            try:
                webbrowser.open(f"http://{self.host}:{self.port}", new=1, autoraise=True)
            except Exception:
                pass

        with contextlib.suppress(Exception):
            server.serve_forever(poll_interval=0.2)

