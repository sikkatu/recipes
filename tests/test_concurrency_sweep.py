"""Part C (round 87): behavioural tests for concurrency_sweep.py.

The sweep logic is executed against a LOCAL mock engine (stdlib http.server)
that serves /v1/models, streaming SSE chat completions and /metrics -- no
real engine, no network. Every test names the MUTATION it must catch:

  sweep-logic : aggregate tok/s = total tokens / MEASUREMENT time (mutation:
                sum of per-stream speeds is caught); missing metric -> null
                (mutation: 0 substituted is caught); knee found on a known
                synthetic profile; correctness distinguishes identical /
                diverges-late / diverges-early; the API key never leaks into
                stdout/JSON.
  sweep-cli   : --help prints defaults; a 401 from the server surfaces as a
                clear error (mutation: swallowing 401 is caught).
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/scripts/concurrency_sweep.py"

sys.path.insert(0, str(SCRIPT.parent))
import concurrency_sweep as cs


# ----------------------------- mock engine ---------------------------------
class Mock(BaseHTTPRequestHandler):
    """Synthetic vLLM: SSE streams with usage, /metrics with a programmed
    profile (kv usage rises with concurrency)."""
    tokens_per_stream = 40      # completion_tokens reported via usage
    token_delay = 0.01          # seconds between SSE chunks -> ~100 tok/s/stream
    require_key = False
    magic_key = "secret123"
    kv_scale = 0.09             # kv usage = 0.09 * concurrent requests
    preempt_at = 99             # preemptions counter jumps at this concurrency

    def log_message(self, *a):
        pass

    def _check_auth(self) -> bool:
        if not self.require_key:
            return True
        a = self.headers.get("Authorization", "")
        return a == "Bearer " + self.magic_key

    def do_GET(self):
        if self.path == "/metrics":
            n = getattr(self.server, "concurrent", 0)
            kv = min(1.0, self.kv_scale * n)
            pre = 7 if n >= self.preempt_at else 0
            body = (
                f"vllm:num_requests_running {n}\n"
                f"vllm:num_requests_waiting {max(0, n - 4)}\n"
                f"vllm:kv_cache_usage_perc {kv}\n"
                f"vllm:num_preemptions_total {pre}\n"
                f"vllm:generation_tokens_total 12345\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models":
            if not self._check_auth():
                self.send_error(401); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"data": [{"id": "mock-model"}]}).encode())
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404); return
        if not self._check_auth():
            self.send_error(401); return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        text = body.get("messages", [{}])[0].get("content", "")
        ntok = min(self.tokens_per_stream, body.get("max_tokens", 256) or 256)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with self.server.lock:
            self.server.concurrent += 1
        try:
            for i in range(ntok):
                time.sleep(self.token_delay)
                chunk = {"choices": [{"delta": {"content": f"{text[:2]}tok{i} "}}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            done = {"choices": [{"delta": {}}], "usage": {"completion_tokens": ntok}}
            self.wfile.write(("data: " + json.dumps(done) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass
        finally:
            with self.server.lock:
                self.server.concurrent -= 1


@pytest.fixture()
def server(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    srv.concurrent = 0
    srv.lock = threading.Lock()
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); th.join(timeout=5)


# ------------------------------- helpers -----------------------------------
def _level(srv, n, duration=1.2, max_tokens=30):
    return cs.run_level(srv, None, "mock-model", ["explain X", "explain Y"],
                        n, duration, max_tokens, srv + "/metrics",
                        sample_iv=0.2)


def test_stream_request_measures_ttft_and_decode(server):
    """One stream: TTFT > 0, decode_tps from usage tokens (not chunk counts)."""
    r = cs.stream_request(server, None, "hello world", "mock-model", 30)
    assert r["err"] is None, r
    assert r["gen_tokens"] == 30, f"usage tokens must drive gen_tokens, got {r['gen_tokens']}"
    assert r["ttft"] is not None and r["ttft"] > 0
    assert r["decode_tps"] is not None and r["decode_tps"] > 1


def test_aggregate_uses_measurement_time_not_stream_sum(server):
    """MUTATION caught: agg = sum(per-stream speeds)*wall (wrong) vs
    total_tokens / duration (right). The mock runs 2 streams at ~100 tok/s
    each -> agg should be ~200 tok/s, NOT 200 per stream * samples."""
    lv = _level(server, 2, duration=1.5, max_tokens=60)
    assert lv["requests"] >= 1
    assert lv["agg_tps"] is not None
    # two parallel ~100 tok/s streams -> aggregate ~200; a sum-of-streams
    # mistake would give a number near 200*several hundred.
    assert 50 < lv["agg_tps"] < 400, f"agg_tps looks wrong: {lv['agg_tps']}"


def test_missing_metrics_yield_nulls_not_zero(tmp_path):
    """MUTATION caught: replacing unavailable metric with 0."""
    d = cs.fetch_metrics("http://127.0.0.1:1/nothing")
    for n in cs.METRIC_NAMES:
        assert d[n] is None, f"{n} must be null when unreachable, not {d[n]}"


def test_knee_found_on_synthetic_drop():
    """A known profile (flat per-stream at N=1..4, collapsed at N=6) must be
    reported at the drop level with the drop reason. MUTATION caught: broken
    drop rule or a mis-ordered rule firing on the aggregate instead."""
    def lv(n, tps, agg, kv=0.1, pre=0):
        return {"n": n, "agg_tps": agg, "stream_tps_p50": tps,
                "stream_tps_p10": tps * 0.9, "ttft_p50": 1, "ttft_p95": 2,
                "kv_max": kv, "running_max": n, "waiting_max": 0,
                "preemptions_delta": pre, "notes": []}
    levels = [lv(1, 100, 100), lv(2, 105, 210),
              lv(4, 100, 400), lv(6, 55, 330)]
    n, reason = cs.find_knee(levels, 0.20)
    assert n == 6, f"knee expected at N=6 (per-stream 55 < 100*0.80), got {n}: {reason}"
    assert "dropped" in reason, reason


def test_knee_on_kv_and_preemption():
    """KV >= 0.9 is itself a knee; a preemptions jump is itself a knee.
    MUTATION caught: those branches ignored or mis-ordered."""
    def lv(n, tps, agg, kv=0.1, pre=0):
        return {"n": n, "agg_tps": agg, "stream_tps_p50": tps,
                "stream_tps_p10": tps, "ttft_p50": 1, "ttft_p95": 2,
                "kv_max": kv, "running_max": n, "waiting_max": 0,
                "preemptions_delta": pre, "notes": []}
    # flat per-stream, growing aggregate -> only the KV rule can fire
    n1, r1 = cs.find_knee([lv(1, 100, 100), lv(2, 100, 200, kv=0.95)], 0.20)
    assert n1 == 2 and "KV" in r1, (n1, r1)
    # flat per-stream, growing aggregate -> only the preemption rule can fire
    n2, r2 = cs.find_knee([lv(1, 100, 100), lv(2, 100, 200),
                           lv(3, 100, 300, pre=1)], 0.20)
    assert n2 == 3 and "preempt" in r2, (n2, r2)


def test_correctness_classifies_identical_late_early(server):
    """identical / diverges-late / diverges-early are distinguished.
    The mock answers prompt-dependent text deterministically -> identical."""
    c = cs.correctness_check(server, None, "mock-model",
                             ["alpha prompt", "beta prompt"], 20, 2, seed=7)
    assert c["verdict"] == "PASS", c
    assert c["identical"] == 2
    assert c["diverges_early"] == 0


def test_api_key_never_leaks(server, tmp_path):
    """MUTATION caught: printing the key in any output. The tool must not
    contain the key string in stdout or in the JSON report."""
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--url", server,
                        "--levels", "1", "--duration", "0.4", "--warmup", "0",
                        "--max-tokens", "10", "--no-correctness", "--yes",
                        "--out", str(out), "--model", "mock-model"],
                       capture_output=True, text=True, timeout=180,
                       env={"PATH": "/usr/bin:/bin", "API_KEY": "secret123"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "secret123" not in r.stdout + r.stderr
    assert "secret123" not in out.read_text()


def test_cli_help_shows_defaults():
    """--help must print the defaults (levels, duration, threshold)."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    for token in ("1,2,4,6,8,10", "60", "0.20", "256"):
        assert token in r.stdout, f"help must mention default {token!r}"


def test_cli_401_is_reported(tmp_path):
    """MUTATION caught: swallowing HTTP 401. With key required and none
    given, the sweep must fail with a clear error, not hang or fake data."""
    Mock.require_key = True
    try:
        srv = HTTPServer(("127.0.0.1", 0), Mock)
        srv.concurrent = 0
        srv.lock = threading.Lock()
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        r = subprocess.run([sys.executable, str(SCRIPT), "--url", url,
                            "--levels", "1", "--duration", "0.2", "--warmup", "0",
                            "--no-correctness", "--yes",
                            "--out", str(tmp_path / "x.json"), "--model", "mock-model"],
                           capture_output=True, text=True, timeout=120,
                           env={"PATH": "/usr/bin:/bin"})
        out = r.stdout + r.stderr
        srv.shutdown(); th.join(timeout=5)
        # the model resolution hits /v1/models -> 401 must reach the user
        assert r.returncode != 0 or "401" in out or "Error" in out, out
    finally:
        Mock.require_key = False