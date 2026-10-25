#!/usr/bin/env python3
"""bench_load.py -- per-request TTFT / decode-speed benchmark (hot/cold split base).

Called by scripts/load-test.sh; can also run standalone:

    python3 scripts/bench_load.py <rounds> <url> [key_file]

For every request it uses streaming (stream=true) so that:
  TTFT    = time to the first generated token (prompt processing + queue wait)
  decode  = (n_gen - 1) / (t_last_token - t_first_token)  -- pure decode speed,
            separated from prompt processing, exactly as serving backends report it

Output: per-request rows (topic, round, gen_tok, ttft, decode tok/s),
per-topic averages and an OVERALL row. Pure stdlib (urllib), no dependencies.
"""
import json
import sys
import time
import urllib.request

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:30007"
KEY_FILE = sys.argv[3] if len(sys.argv) > 3 else "live/api_key"
try:
    KEY = open(KEY_FILE).read().strip()
except OSError:
    sys.exit(f"bench_load: cannot read API key file {KEY_FILE}")

TOPICS = [
    ("geography",   "What is the capital of Australia? Answer in one sentence.", 80),
    ("math",        "Calculate 247 * 38 step by step.", 200),
    ("code",        "Write a Python one-liner to sum a list of integers.", 100),
    ("logic",       "If some A are B, and all B are C, can some A be C? Explain.", 200),
    ("history",     "Name three causes of the French Revolution in one paragraph.", 200),
    ("physics",     "Explain Newton's second law in one sentence.", 100),
    ("translation", "Translate to German: The weather is nice today.", 80),
    ("creative",    "Write one haiku about the ocean.", 120),
    ("science",     "What is photosynthesis? One paragraph.", 200),
    ("moe",         "What is a Mixture of Experts layer? One paragraph.", 200),
]

def ask(topic, prompt, mt):
    """One streaming request. Returns (topic, n_gen, ttft, decode_tok_s, wall)."""
    body = json.dumps({"model": "mimo-pro",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mt, "temperature": 0.7,
                       "stream": True}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    n_gen = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or ""
                if piece:
                    n_gen += 1
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
    wall = time.time() - t0
    if t_first is None:
        return topic, 0, wall, 0.0, wall
    ttft = t_first - t0
    span = (t_last or t_first) - t_first
    decode = (n_gen - 1) / span if span > 0 and n_gen > 1 else 0.0
    return topic, n_gen, ttft, decode, wall

print(f"{'topic':<14} {'round':<6} {'gen_tok':<8} {'ttft_s':<8} {'dec_tps':<8} {'wall_s':<7}")
dec_all, ttft_all = {}, {}
for r in range(1, ROUNDS + 1):
    for topic, prompt, mt in TOPICS:
        try:
            t, gen, ttft, dec, wall = ask(topic, prompt, mt)
            dec_all.setdefault(t, []).append(dec)
            ttft_all.setdefault(t, []).append(ttft)
            print(f"{t:<14} {r:<6} {gen:<8} {ttft:<8.2f} {dec:<8.1f} {wall:<7.1f}")
        except Exception as e:
            print(f"{topic:<14} {r:<6} ERROR {e}")

print("\n=== averages per topic (decode tok/s | TTFT s) ===")
for topic in sorted(dec_all):
    ds, ts = dec_all[topic], ttft_all[topic]
    print(f"{topic:<14} {sum(ds)/len(ds):.1f} tok/s | ttft {sum(ts)/len(ts):.2f}s  (n={len(ds)})")
ds_overall = [d for v in dec_all.values() for d in v]
ts_overall = [t for v in ttft_all.values() for t in v]
if ds_overall:
    print(f"{'OVERALL':<14} {sum(ds_overall)/len(ds_overall):.1f} tok/s | ttft "
          f"{sum(ts_overall)/len(ts_overall):.2f}s  (n={len(ds_overall)})")

# machine-readable results so runs can be compared later (dynamic vs static, x8 vs x16, ...)
import datetime
import os

results = {
    "when": datetime.datetime.now().isoformat(timespec="seconds"),
    "url": URL,
    "rounds": ROUNDS,
    "n_requests": len(ds_overall),
    "decode_tps_overall": round(sum(ds_overall) / len(ds_overall), 2) if ds_overall else None,
    "ttft_s_overall": round(sum(ts_overall) / len(ts_overall), 3) if ts_overall else None,
    "per_topic": {
        topic: {
            "decode_tps": round(sum(dec_all[topic]) / len(dec_all[topic]), 2),
            "ttft_s": round(sum(ttft_all[topic]) / len(ttft_all[topic]), 3),
            "n": len(dec_all[topic]),
        } for topic in sorted(dec_all)
    },
}
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "bench_results.json")
try:
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved: {outpath}")
except OSError as e:
    print(f"\nNOTE: could not save results ({e})")