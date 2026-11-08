"""Round 66 regression: the reviewer's repro-round65.sh as a pytest suite.

Scenarios (from the reviewer's repro script):
  A. check-tp4-api.sh with a key -> exit 0
  B. manage.sh status with the right key -> "reachable (authenticated)"
  C. manage.sh status with a wrong key -> "REJECTED"
  D. manage.sh status with a dead server -> exit 0 (was exit 7 before round 65)
  E. no key vs a no-auth server -> exit 0
  F. kvp_curl must deliver keys with special characters intact
     (escaped quotes/backslashes reach the server byte-for-byte)

Needs bash + curl + python3 only (no docker/GPU -- docker is stubbed).
The HTTP servers run on ephemeral ports inside the test process.
"""
from __future__ import annotations
import http.server
import os
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parent.parent
SCRIPTS = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "scripts"
KV_REPO = next(
    (c for c in (
        RECIPES / "kv-persistence",
        Path.home() / "kv-persistence",
        RECIPES.parent / "kv-persistence",
    ) if c.is_dir()),
    None,
)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(expect_key: str, body: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        def _go(self):
            got = self.headers.get("Authorization", "")
            ok = (expect_key == "") or (got == f"Bearer {expect_key}")
            self.send_response(200 if ok else 401)
            self.end_headers()
            self.wfile.write(body)

        do_GET = _go

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            self._go()

        def log_message(self, *a):
            pass

    return H


class _Live:
    """A running test HTTP server on an ephemeral port."""

    def __init__(self, expect_key: str, body: bytes):
        self.httpd = _Server(("127.0.0.1", 0), _make_handler(expect_key, body))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def auth_server():
    s = _Live("secretkey123", b'{"data":[{"id":"mimo-pro"}]}')
    yield s
    s.close()


@pytest.fixture()
def noauth_server():
    s = _Live("", b'{"data":[{"id":"mimo-pro"}]}')
    yield s
    s.close()


@pytest.fixture()
def echo_server():
    """Echoes the Authorization header back in the body -- scenario F."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(("AUTH=[%s]" % self.headers.get("Authorization", "")).encode())

        def log_message(self, *a):
            pass

    httpd = _Server(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def stub_env(tmp_path):
    """Minimal env: stub docker on PATH, isolated HOME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n")
    docker.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(home)
    return env


def test_a_check_api_with_key(auth_server, stub_env):
    r = subprocess.run(
        ["bash", str(SCRIPTS / "check-tp4-api.sh"), f"http://127.0.0.1:{auth_server.port}"],
        env={**stub_env, "API_KEY": "secretkey123"}, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_b_status_with_key(auth_server, stub_env):
    r = subprocess.run(
        ["bash", str(SCRIPTS / "manage.sh"), "status"],
        env={**stub_env, "API_KEY": "secretkey123",
             "BASE": f"http://127.0.0.1:{auth_server.port}",
             "PORT": str(auth_server.port)},
        capture_output=True, text=True, timeout=30,
    )
    assert "reachable (authenticated)" in r.stdout + r.stderr


def test_c_status_wrong_key(auth_server, stub_env):
    r = subprocess.run(
        ["bash", str(SCRIPTS / "manage.sh"), "status"],
        env={**stub_env, "API_KEY": "wrong",
             "BASE": f"http://127.0.0.1:{auth_server.port}",
             "PORT": str(auth_server.port)},
        capture_output=True, text=True, timeout=30,
    )
    assert "REJECTED" in r.stdout + r.stderr


def test_d_status_dead_server_exits_zero(stub_env):
    r = subprocess.run(
        ["bash", str(SCRIPTS / "manage.sh"), "status"],
        env={**stub_env, "API_KEY": "secretkey123",
             "BASE": "http://127.0.0.1:38999", "PORT": "38999"},
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, "status must not die on a dead server (set -e bug)"


def test_e_no_key_vs_noauth_server(noauth_server, stub_env):
    env = {**stub_env, "API_KEY": ""}
    r = subprocess.run(
        ["bash", str(SCRIPTS / "check-tp4-api.sh"), f"http://127.0.0.1:{noauth_server.port}"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("key", ["plain123", 'abc"def', "abc\\def", "a b#c"])
def test_f_kvp_curl_delivers_special_keys_intact(key, echo_server):
    """Round 66: the curl config file parses quoted values -- a raw " or \
    inside the key used to be silently mangled. kvp_curl must escape both."""
    if KV_REPO is None or not (KV_REPO / "scripts" / "env.sh").is_file():
        pytest.skip("kv-persistence checkout not found")
    # Pass the key as a POSITIONAL argument (repr() in single quotes would
    # keep the backslash escapes literal -- exactly the repro-script style).
    script = (
        f'. "{KV_REPO}/scripts/env.sh"; '
        'API_KEY="$1"; '
        f'kvp_curl -s http://127.0.0.1:{echo_server}/'
    )
    r = subprocess.run(["bash", "-c", script, "_", key], capture_output=True, text=True, timeout=30)
    assert r.stdout.strip() == f"AUTH=[Bearer {key}]", (
        f"key {key!r} was mangled in transit: {r.stdout!r}")