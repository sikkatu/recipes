#!/bin/bash
# check-prefix-cache.sh — verify prefix caching with REAL measurements.
# Run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/ (server must be running).
#
# Why not cached_tokens in the API response: this vLLM build returns
# prompt_tokens_details:null, so the API field is not a reliable signal here.
# The ground truth is the engine's own Prometheus counters (vllm:prefix_cache_hits /
# vllm:prefix_cache_queries) exposed on /metrics. This script reads those, plus
# decode speed (tok/s) per call.
KEY="$(cat live/api_key)"
BASE="http://127.0.0.1:30007"

python3 - <<'PYEOF'
import json, time, urllib.request

KEY = open("live/api_key").read().strip()
BASE = "http://127.0.0.1:30007"

def metrics():
    """Return (queries, hits) from the engine's Prometheus counters."""
    req = urllib.request.Request(BASE + "/metrics")
    req.add_header("Authorization", "Bearer " + KEY)
    try:
        text = urllib.request.urlopen(req, timeout=5).read().decode()
    except Exception:
        return None, None
    q = h = 0
    for line in text.splitlines():
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

def ask(prompt, max_tokens=50):
    body = json.dumps({
        "model": "mimo-pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req).read())
    dt = time.time() - t0
    u = resp["usage"]
    gen = u.get("completion_tokens", 0)
    return u["prompt_tokens"], gen, dt

# long shared prefix: same ~2000-token prologue on every call
para = "The history of computing spans thousands of years, from the abacus to modern AI. " * 120

q0, h0 = metrics()
if q0 is not None:
    print(f"metrics before: queries={q0:.0f}  hits={h0:.0f}")
else:
    print("metrics before: /metrics unavailable (need --enable-metrics in the launcher)")

results = []
for i in range(1, 4):
    prompt = para + f"\n\nQuestion {i}: what is the capital of Japan?"
    pt, gt, dt = ask(prompt)
    speed = gt / dt if dt > 0 else 0
    results.append((pt, gt, dt, speed))
    print(f"run {i}: prompt={pt} tok, generated={gt} tok, {dt:.1f}s, {speed:.1f} tok/s")

q1, h1 = metrics()
if q0 is not None and q1 is not None:
    dq = q1 - q0
    dh = h1 - h0
    rate = 100.0 * dh / dq if dq else 0
    print(f"\nmetrics after: queries={q1:.0f}  hits={h1:.0f}")
    print(f"this session: {dh:.0f} cache hits / {dq:.0f} queries = {rate:.1f}% hit rate")
    if dh > 0:
        print("PREFIX CACHING WORKS - the engine reports cache hits.")
    else:
        print("No cache hits recorded. Check --enable-prefix-caching and that requests share a prefix.")

if len(results) >= 2:
    warm, steady = results[0], results[-1]
    if steady[3] > 0:
        print(f"speed: warmup {warm[3]:.1f} tok/s -> steady {steady[3]:.1f} tok/s "
              f"(first call includes torch.compile warmup)")
PYEOF