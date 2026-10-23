#!/bin/bash
# load-test.sh — mixed-traffic load run: measures tok/s per topic and the hot-bank hit
# rate over time. Run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/ while the server
# (hotsplit, dynamic) is up.
#
# Usage: bash scripts/load-test.sh [rounds]     (default 5 rounds x 10 topics = 50 requests)
#
# After the run it prints the summary AND the dynamic hit-rate lines from the container
# log so you can see how hit-rate evolves as the planner adapts.

ROUNDS="${1:-5}"
KEY="$(cat live/api_key)"
URL="http://127.0.0.1:30007"
CONTAINER="${CONTAINER:-mimo-pro-tp4-hotsplit}"

python3 - "$ROUNDS" "$URL" <<'PYEOF'
import json, sys, time, urllib.request

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
KEY = open("live/api_key").read().strip()
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:30007"

TOPICS = [
    ("geography",  "What is the capital of Australia? Answer in one sentence.", 80),
    ("math",       "Calculate 247 * 38 step by step.", 200),
    ("code",       "Write a Python one-liner to sum a list of integers.", 100),
    ("logic",      "If some A are B, and all B are C, can some A be C? Explain.", 200),
    ("history",    "Name three causes of the French Revolution in one paragraph.", 200),
    ("physics",    "Explain Newton's second law in one sentence.", 100),
    ("translation","Translate to German: The weather is nice today.", 80),
    ("creative",   "Write one haiku about the ocean.", 120),
    ("science",    "What is photosynthesis? One paragraph.", 200),
    ("moe",        "What is a Mixture of Experts layer? One paragraph.", 200),
]

def ask(topic, prompt, mt):
    body = json.dumps({"model": "mimo-pro",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mt, "temperature": 0.7}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    u = resp["usage"]
    gen = u.get("completion_tokens", 0)
    return topic, gen, dt, gen / dt if dt > 0 else 0

print(f"{'topic':<14} {'round':<6} {'gen_tok':<8} {'time':<7} {'tok/s':<7}")
all_speeds = {}
for r in range(1, ROUNDS + 1):
    for topic, prompt, mt in TOPICS:
        try:
            t, gen, dt, sp = ask(topic, prompt, mt)
            all_speeds.setdefault(topic, []).append(sp)
            print(f"{t:<14} {r:<6} {gen:<8} {dt:<7.1f} {sp:<7.1f}")
        except Exception as e:
            print(f"{topic:<14} {r:<6} ERROR {e}")

print("\n=== average tok/s per topic ===")
for topic, speeds in sorted(all_speeds.items()):
    print(f"{topic:<14} {sum(speeds)/len(speeds):.1f} tok/s  (n={len(speeds)})")
overall = [s for v in all_speeds.values() for s in v]
if overall:
    print(f"{'OVERALL':<14} {sum(overall)/len(overall):.1f} tok/s  (n={len(overall)})")
PYEOF

echo
echo "=== dynamic hit-rate evolution (from container log) ==="
docker logs "$CONTAINER" 2>&1 | grep 'hit-rate' | tail -10
echo
echo "=== dynamic updates so far ==="
docker logs "$CONTAINER" 2>&1 | grep 'hotsplit dynamic update' | tail -6