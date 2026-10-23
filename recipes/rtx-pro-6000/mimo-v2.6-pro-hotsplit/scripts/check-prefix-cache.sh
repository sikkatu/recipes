#!/bin/bash
# check-prefix-cache.sh — verify prefix caching with REAL measurements.
# Run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/ (server must be running).
#
# Two independent signals, tried in order:
#   1. /metrics Prometheus counters (vllm:prefix_cache_hits / vllm:prefix_cache_queries)
#      — the ground truth when the endpoint is available.
#   2. Engine log lines "Prefix cache hit rate: X%" — printed by vLLM's stats logger
#      periodically under load; works when /metrics is not exposed.
# The OpenAI API field prompt_tokens_details.cached_tokens is NOT used: this vLLM
# build returns prompt_tokens_details:null, so it is never populated.
# Also prints decode speed (tok/s) per call — the first call includes torch.compile
# warmup, so compare run 2+ against each other.

python3 - <<'PYEOF'
import json, time, urllib.request

KEY = open("live/api_key").read().strip()
BASE = "http://127.0.0.1:30007"
CONTAINER = "mimo-pro-tp4-hotsplit"

def metrics():
    """(queries, hits) from Prometheus counters, or (None, None) if /metrics is not exposed."""
    req = urllib.request.Request(BASE + "/metrics")
    req.add_header("Authorization", "Bearer " + KEY)
    try:
        text = urllib.request.urlopen(req, timeout=5).read().decode()
    except Exception:
        return None, None
    q = h = 0
    for line in text.splitlines():
        if "_created" in line:
            continue
        if line.startswith("vllm:prefix_cache_queries"):
            try:
                q += float(line.rsplit(None, 1)[-1])
            except ValueError:
                pass
        elif line.startswith("vllm:prefix_cache_hits"):
            try:
                h += float(line.rsplit(None, 1)[-1])
            except ValueError:
                pass
    return q, h

def log_hit_rates():
    """Last 'Prefix cache hit rate' lines from the container log (fallback signal)."""
    import subprocess
    try:
        out = subprocess.run(["docker", "logs", "--tail", "300", CONTAINER],
                             capture_output=True, text=True, timeout=10).stderr
    except Exception:
        return []
    return [ln for ln in out.splitlines() if "Prefix cache hit rate" in ln]

def ask(prompt, max_tokens=50):
    body = json.dumps({"model": "mimo-pro",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req).read())
    dt = time.time() - t0
    u = resp["usage"]
    return u["prompt_tokens"], u.get("completion_tokens", 0), dt

# long shared prefix: same ~2000-token prologue on every call -> cacheable blocks
para = "The history of computing spans thousands of years, from the abacus to modern AI. " * 120

q0, h0 = metrics()
print(f"metrics before: queries={q0} hits={h0}" if q0 is not None
      else "metrics before: /metrics not exposed (log-based check below)")

results = []
for i in range(1, 4):
    prompt = para + f"\n\nQuestion {i}: what is the capital of Japan?"
    pt, gt, dt = ask(prompt)
    sp = gt / dt if dt > 0 else 0
    results.append((pt, gt, dt, sp))
    print(f"run {i}: prompt={pt} tok, generated={gt} tok, {dt:.1f}s, {sp:.1f} tok/s")

q1, h1 = metrics()
if q0 is not None and q1 is not None:
    dq, dh = q1 - q0, h1 - h0
    rate = 100.0 * dh / dq if dq else 0
    print(f"\nmetrics after: queries={q1} hits={h1}")
    print(f"this session: {dh:.0f} cache-hit tokens / {dq:.0f} queried tokens = {rate:.1f}% hit rate")
    if dh > 0:
        print("PREFIX CACHING WORKS (engine counters).")
    else:
        print("No cache hits in counters. Check that requests share a prefix and the run is not too short.")
else:
    lines = log_hit_rates()
    if lines:
        print("\nengine log (fallback signal):")
        for ln in lines[-3:]:
            print("  " + ln.strip()[-90:])
        print("PREFIX CACHING: see 'Prefix cache hit rate' above - non-zero on repeated prefixes.")
    else:
        print("\nNeither /metrics nor log lines available yet. Generate more traffic and re-run.")

if len(results) >= 2:
    warm, steady = results[0], results[-1]
    print(f"speed: warmup {warm[3]:.1f} tok/s -> steady {steady[3]:.1f} tok/s "
          f"(first call includes torch.compile warmup)")
PYEOF