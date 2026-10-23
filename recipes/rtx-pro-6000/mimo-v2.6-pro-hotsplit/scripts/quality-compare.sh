#!/bin/bash
# quality-compare.sh — 5 identical queries against the hotsplit server and the UVA baseline,
# with the answers saved side by side. Run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/.
#
# Usage: bash scripts/quality-compare.sh [hotsplit_url] [baseline_url]
#   defaults: http://127.0.0.1:30007 (hotsplit) and http://127.0.0.1:30007 (same port —
#   stop one server before starting the other; the script prints what to do).
#
# This answers "do the hot/cold swaps degrade output quality?" — run it once against
# hotsplit (HOTSPLIT_DYNAMIC=1), save the output, then once against the UVA baseline
# (run-tp4-uva.sh) and diff the two files.

KEY="$(cat live/api_key)"
URL="${1:-http://127.0.0.1:30007}"
OUT="${2:-/tmp/quality_$(date +%H%M%S).txt}"

python3 - "$URL" "$OUT" <<'PYEOF'
import json, sys, urllib.request

KEY = open("live/api_key").read().strip()
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30007"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/quality.txt"

QUERIES = [
    ("What is the capital of France? Answer briefly.", 100),
    ("What is 17 * 23 + 45? Show your work.", 150),
    ("Explain what a Mixture of Experts (MoE) layer is in one paragraph.", 200),
    ("A train travels at 60 km/h for 2.5 hours. How far does it go?", 150),
    ("Напиши короткое стихотворение о весне (4 строки).", 200),
]

lines = []
for q, mt in QUERIES:
    body = json.dumps({
        "model": "mimo-pro",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": mt,
        "temperature": 0
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        m = resp["choices"][0]["message"]
        content = m.get("content") or ""
        reasoning = m.get("reasoning") or ""
        answer = content if content.strip() else f"(content empty, reasoning: {reasoning[:200]})"
        finish = resp["choices"][0].get("finish_reason", "?")
    except Exception as e:
        answer = f"ERROR: {e}"
        finish = "error"
    lines.append(f"Q: {q}\nA: {answer}\n   [finish_reason={finish}]\n")

text = "\n".join(lines)
with open(OUT, "w") as f:
    f.write(text)
print(text)
print(f"\nSaved to {OUT}")
print("Run this script twice (once per server), then: diff /tmp/quality_*.txt")
PYEOF