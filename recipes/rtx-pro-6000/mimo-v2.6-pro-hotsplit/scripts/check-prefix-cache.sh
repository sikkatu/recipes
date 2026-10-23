#!/bin/bash
# PREFIX CACHING TEST — run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/
# Sends the same long-prefix request twice and shows cached_tokens.
# "The long text" = a shared prefix repeated to fill ~2000 tokens: the caching
# system hashes KV blocks of the prefix, so run 2 should reuse them.
KEY="$(cat live/api_key)"
python3 - <<'PYEOF'
import json, urllib.request, time

KEY = open("live/api_key").read().strip()
URL = "http://127.0.0.1:30007/v1/chat/completions"

# Shared prefix: a factual paragraph repeated to fill ~2000 tokens.
# Both requests start with this exact text -> the KV blocks are cacheable.
para = "The history of computing spans thousands of years, from the abacus to modern AI. "
prompt = para * 120 + "\n\nWhat is the capital of Japan?"

def ask():
    body = json.dumps({
        "model": "mimo-pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
        "temperature": 0
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": "Bearer " + KEY,
        "Content-Type": "application/json"
    })
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req).read())
    dt = time.time() - t0
    u = resp["usage"]
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    return u["prompt_tokens"], cached, dt

p1, c1, t1 = ask()
print(f"run 1: prompt_tokens={p1}  cached_tokens={c1}  time={t1:.1f}s")
p2, c2, t2 = ask()
print(f"run 2: prompt_tokens={p2}  cached_tokens={c2}  time={t2:.1f}s")
if c2 > 0:
    print(f"\nPREFIX CACHING WORKS: {c2}/{p2} tokens from cache ({100*c2//p2}%), "
          f"second call {t1/t2:.1f}x faster")
else:
    print("\nNo cache hit on run 2 — prefix caching may be disabled or the prompt was not reused")
PYEOF