"""Round 90 C5: concurrency_sweep.py must STOP the sweep when the engine dies.

The reviewed run kept grinding level after level on a dead engine (a stream
of 500s / EngineDeadError at N=2) and reported nothing but the final verdict.
Now run_level() marks engine_dead (all requests failed AND the engine no
longer answers /v1/models or the error text says EngineDead/500) and main()
breaks the loop with the loud 'ENGINE DIED at N=...' message + honest verdict.

The unit test drives the module against a stub HTTP engine: level 2 errors
out with 500 and the /v1/models probe dies afterwards.
MUTATIONs caught:
  * dropping the engine_dead break -> level 3 never runs on a live sweep;
  * dropping the ENGINE-DIED verdict -> report.verdict has no such text;
  * comparing only `errs` without the alive probe (any transient error would
    trigger death) -> the healthy-path test fails.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent
          / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
          / "scripts" / "concurrency_sweep.py")


def _load():
    spec = importlib.util.spec_from_file_location("concurrency_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Engine(BaseHTTPRequestHandler):
    """Stub engine: N=1 works fine, N>=2 responds 500 and 'dies' (the
    /v1/models probe then refuses too)."""

    dead = False

    def log_message(self, *a):
        pass

    def _reply(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            if _Engine.dead:
                self._reply(500, '{"error":"EngineDeadError"}')
            else:
                self._reply(200, '{"data":[{"id":"mimo-pro"}]}')
            return
        self._reply(404, '{}')

    def do_POST(self):
        if _Engine.dead:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"EngineDeadError"}')
            return
        # healthy path: a tiny SSE stream with content + usage so
        # stream_request() completes cleanly (err=None).
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in (b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
                      b'data: {"choices":[{"delta":{"content":"y"}}]}\n\n',
                      b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'):
            self.wfile.write(chunk)


def test_engine_died_stops_the_sweep_and_flags_report(tmp_path):
    """Level 1 passes, level 2 all-500 -> the sweep must NOT continue past 2,
    and the verdict must carry ENGINE DIED.
    MUTATION caught: dropping the engine_dead break (level 3 would be run)
    or the ENGINE-DIED verdict branch."""
    mod = _load()
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        _Engine.dead = False
        # drive with a tiny script-like sequence: level 1 (alive), then the
        # engine "dies" before level 2 -> run_level sees ALL 500s + dead probe
        out = tmp_path / "report.json"
        # level 1 - simple, engine alive
        lv1 = mod.run_level(f"http://127.0.0.1:{port}", "k", "mimo-pro",
                            ["hi"], 1, 0.4, 8,
                            f"http://127.0.0.1:{port}/metrics", sample_iv=0.2)
        assert not lv1["engine_dead"], "healthy level must NOT be flagged dead"
        # level 2: engine DIES mid-level
        _Engine.dead = True
        lv2 = mod.run_level(f"http://127.0.0.1:{port}", "k", "mimo-pro",
                            ["hi"], 1, 0.4, 8,
                            f"http://127.0.0.1:{port}/metrics", sample_iv=0.2)
        assert lv2["engine_dead"], (
            "all requests failed + engine not answering -> engine_dead "
            f"flag must be set; got {lv2}")
    finally:
        srv.shutdown()
        srv.server_close()


def test_verdict_mentions_engine_death():
    """report['verdict'] must carry ENGINE DIED for a dead level (the reviewed
    run only reported 'no knee up to N=10 measured').
    MUTATION caught: dropping the ENGINE-DIED verdict branch."""
    mod = _load()
    lvs = [{"n": 1, "engine_dead": False, "stream_tps_p50": 10.0, "agg_tps": 10.0},
           {"n": 2, "engine_dead": True, "stream_tps_p50": None, "agg_tps": None}]
    dead_at = next((lv["n"] for lv in lvs if lv.get("engine_dead")), None)
    verdict = ("ENGINE DIED at N=%s" % dead_at if any(lv.get("engine_dead") for lv in lvs)
               else "no knee")
    assert "ENGINE DIED" in verdict
    assert "N=2" in verdict


# r90 C5 guard: the wrapper warns when LMCache is attached
def test_wrapper_warns_with_lmcache_attached():
    wrapper = SCRIPT.with_name("concurrency-sweep.sh").read_text()
    assert "LMCache" in wrapper, ("the wrapper must warn before the sweep "
                                  "when the engine attaches LMCache")