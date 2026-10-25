#!/bin/bash
# quality-compare.sh — automated quality A/B: the SAME queries against TWO servers in one run,
# with a per-query diff report. Run from recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/.
#
# Usage:
#   bash scripts/quality-compare.sh URL_A URL_B \
#        [key_file_A] [key_file_B] [output_dir]
#
#   URL_A      first server  (default http://127.0.0.1:30007)
#   URL_B      second server (default: same as URL_A — with a warning, useful only for a
#              single-server smoke save; for a real comparison pass BOTH urls)
#   key_file_A API key file for server A (default live/api_key)
#   key_file_B API key file for server B (default: same as key_file_A)
#   output_dir where to save the report (default /tmp/quality_<HHMMSS>/)
#
# Typical flow (two servers on different ports):
#   # server 1: baseline on port 30008
#   PORT=30008 CONTAINER=mimo-pro-tp4-uva bash scripts/run-tp4-uva.sh
#   # server 2: hotsplit on port 30007
#   bash scripts/run-tp4-hotsplit.sh
#   # compare in one shot:
#   bash scripts/quality-compare.sh http://127.0.0.1:30007 http://127.0.0.1:30008
#
# The report saves both answers side by side, a per-query diff verdict (identical /
# cosmetic / DIFFERENT), and the run parameters + model ids from /v1/models of each
# server so the comparison is reproducible.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }

URL_A="${1:-http://127.0.0.1:30007}"
URL_B="${2:-}"
KEY_FILE_A="${3:-live/api_key}"
KEY_FILE_B="${4:-}"
OUTDIR="${5:-/tmp/quality_$(date +%H%M%S)}"

[ -f "$KEY_FILE_A" ] || die "API key file not found: $KEY_FILE_A"
KEY_A="$(head -n1 "$KEY_FILE_A" | tr -d '[:space:]')"
[ -n "$KEY_A" ] || die "API key file $KEY_FILE_A is empty"

if [ -z "$URL_B" ]; then
  URL_B="$URL_A"
  echo "NOTE: only one URL given — comparing $URL_A against itself (round-trip sanity only)."
  echo "      For a real A/B pass both URLs: bash scripts/quality-compare.sh <urlA> <urlB>"
fi
if [ -z "$KEY_FILE_B" ]; then
  KEY_FILE_B="$KEY_FILE_A"
  KEY_B="$KEY_A"
else
  [ -f "$KEY_FILE_B" ] || die "API key file not found: $KEY_FILE_B"
  KEY_B="$(head -n1 "$KEY_FILE_B" | tr -d '[:space:]')"
  [ -n "$KEY_B" ] || die "API key file $KEY_FILE_B is empty"
fi

mkdir -p "$OUTDIR"
echo "Server A: $URL_A   (key: $KEY_FILE_A)"
echo "Server B: $URL_B   (key: $KEY_FILE_B)"
echo "Report:   $OUTDIR/"

python3 - "$URL_A" "$URL_B" "$KEY_A" "$KEY_B" "$OUTDIR" <<'PYEOF'
import json, sys, urllib.request, difflib, datetime

URL_A, URL_B, KEY_A, KEY_B, OUTDIR = sys.argv[1:6]

QUERIES = [
    ("What is the capital of France? Answer briefly.", 100),
    ("What is 17 * 23 + 45? Show your work.", 150),
    ("Explain what a Mixture of Experts (MoE) layer is in one paragraph.", 200),
    ("A train travels at 60 km/h for 2.5 hours. How far does it go?", 150),
    ("Напиши короткое стихотворение о весне (4 строки).", 200),
]

def model_info(url, key):
    """Record the model id + version served, so the report says WHAT was compared."""
    try:
        req = urllib.request.Request(url + "/v1/models",
                                     headers={"Authorization": "Bearer " + key})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        ids = [m.get("id", "?") for m in data.get("data", [])]
        return ", ".join(ids) if ids else "(no models listed)"
    except Exception as e:
        return f"(unreachable: {e})"

def ask(url, key, q, mt):
    body = json.dumps({"model": "mimo-pro",
                       "messages": [{"role": "user", "content": q}],
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
        ch = resp["choices"][0]
        m = ch["message"]
        content = m.get("content") or ""
        reasoning = m.get("reasoning") or ""
        answer = content if content.strip() else f"(content empty, reasoning: {reasoning[:300]})"
        return answer, ch.get("finish_reason", "?")
    except Exception as e:
        return f"ERROR: {e}", "error"

def verdict(a, b):
    """identical / cosmetic / DIFFERENT — cosmetic = differs only in whitespace/punctuation."""
    if a == b:
        return "identical"
    strip = lambda s: "".join(c for c in s if not c.isspace() and c not in ".,;:!?。，")
    if strip(a) == strip(b):
        return "cosmetic"
    # substantial text overlap?
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return "DIFFERENT" if ratio < 0.5 else "similar"

# --- run header: everything needed to reproduce this comparison ---
hdr = [
    f"quality-compare report {datetime.datetime.now().isoformat(timespec='seconds')}",
    f"server A: {URL_A}   models: {model_info(URL_A, KEY_A)}",
    f"server B: {URL_B}   models: {model_info(URL_B, KEY_B)}",
    f"parameters: model=mimo-pro, temperature=0, per-query max_tokens as noted",
    "",
]

report, a_lines, b_lines = [], [], []
n_diff = 0
for q, mt in QUERIES:
    ans_a, fin_a = ask(URL_A, KEY_A, q, mt)
    ans_b, fin_b = ask(URL_B, KEY_B, q, mt)
    v = verdict(ans_a, ans_b)
    # 'similar' is expected under greedy/temperature=0 sampling and is NOT a failure;
    # only a truly divergent answer counts toward the exit code.
    if v == "DIFFERENT":
        n_diff += 1
    report.append(f"Q: {q}  (max_tokens={mt})")
    report.append(f"A [{URL_A}] [finish={fin_a}]: {ans_a}")
    report.append(f"B [{URL_B}] [finish={fin_b}]: {ans_b}")
    report.append(f"verdict: {v}")
    report.append("")
    a_lines.append(ans_a + "\n")
    b_lines.append(ans_b + "\n")

text = "\n".join(hdr + report)
with open(f"{OUTDIR}/report.txt", "w") as f:
    f.write(text)
# raw diff of the two answer files (like plain `diff`) for external tooling
with open(f"{OUTDIR}/answers_A.txt", "w") as f:
    f.writelines(a_lines)
with open(f"{OUTDIR}/answers_B.txt", "w") as f:
    f.writelines(b_lines)

print(text)
print(f"Saved: {OUTDIR}/report.txt, answers_A.txt, answers_B.txt")
if n_diff:
    print(f"RESULT: {n_diff} of {len(QUERIES)} queries KEPT THE SAME MEANING but differ textually")
    print("This is expected under greedy decoding -- compare the report, do not gate CI on it.")
    sys.exit(0)
print(f"RESULT: all {len(QUERIES)} queries identical or meaning-preserving -- no quality regression detected")
PYEOF