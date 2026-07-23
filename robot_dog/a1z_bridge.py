#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DirectorX <-> GALAXEA A1Z official control protocol bridge (HTTP -> Unix socket pass-through)

Official a1z server (GALAXEA-A1Z SDK gripper branch a1z/robots/server.py) listens on
Unix socket /tmp/a1z.sock, protocol is JSON with newline ending:
    request:  {"cmd": "<name>", "args": {...}}
    response: {"ok": true, "data": {...}}  or  {"ok": false, "error": "<message>"}
    commands: status | move | gripper | dance | stop | info | estop | release

Browsers cannot connect to Unix sockets directly; this bridge transparently forwards to local HTTP:
    POST http://127.0.0.1:8766/a1z   Body: {"cmd":"move","args":{"joints":[0,60,-60,0,0,0],"speed":0.5}}
    GET  http://127.0.0.1:8766/health  health check

Usage (on the Linux host connected to the robot arm):
    1. Start official service:  python tools/a1zctl serve --with-gripper
    2. Start this bridge:       python a1z_bridge.py            # default 127.0.0.1:8766
                                python a1z_bridge.py --host 0.0.0.0 --port 8766
    3. In DirectorX control bar, switch to A1Z HTTP

Only depends on Python standard library. move/dance are blocking; timeout widened to 75s.
"""
import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_CMDS = {"status", "move", "gripper", "dance", "stop", "info", "estop", "release"}


def rpc(sock_path: str, req: dict, timeout: float) -> dict:
    """Send a JSON-line request to the official server and read response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    line = buf.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    sock_path = "/tmp/a1z.sock"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "a1z_bridge", "sock": self.sock_path})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/a1z":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            cmd = req.get("cmd", "")
            if cmd not in ALLOWED_CMDS:
                raise ValueError(f"unsupported cmd '{cmd}' (allowed: {sorted(ALLOWED_CMDS)})")
            timeout = 75.0 if cmd in {"move", "dance"} else 8.0
            resp = rpc(self.sock_path, {"cmd": cmd, "args": req.get("args", {})}, timeout)
        except FileNotFoundError:
            resp = {"ok": False, "error": f"bridge: official server socket not found ({self.sock_path}) -- start `python tools/a1zctl serve --with-gripper` first"}
        except ConnectionRefusedError:
            resp = {"ok": False, "error": "bridge: official server refused connection -- is `a1zctl serve` running?"}
        except socket.timeout:
            resp = {"ok": False, "error": "bridge: official server response timeout"}
        except Exception as e:
            resp = {"ok": False, "error": f"bridge: {e}"}
        self._json(200, resp)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DirectorX A1Z protocol bridge")
    ap.add_argument("--sock", default="/tmp/a1z.sock", help="Unix socket path of official server")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP listen address (use 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8766, help="HTTP listen port")
    args = ap.parse_args()
    Handler.sock_path = args.sock
    print(f"[a1z_bridge] listening on http://{args.host}:{args.port}  ->  {args.sock}")
    print("[a1z_bridge] Make sure official service is started: python tools/a1zctl serve --with-gripper")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
