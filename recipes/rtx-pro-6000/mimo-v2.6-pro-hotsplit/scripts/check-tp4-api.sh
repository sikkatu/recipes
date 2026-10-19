#!/usr/bin/env bash
# API smoke test for the TP4 server. Sends the API key when one is set; without a key the
# requests are only valid for an ALLOW_NO_AUTH loopback server (a protected API answers 401).
# usage: check-tp4-api.sh [base-url]   (default http://127.0.0.1:30007)
set -Eeuo pipefail
BASE="${1:-${BASE:-http://127.0.0.1:30007}}"
AUTH=()
if [ -n "${API_KEY:-}" ]; then
  AUTH=(-H "Authorization: Bearer $API_KEY")
else
  echo "NOTE: API_KEY is not set. If the server requires auth (the default), these requests" >&2
  echo "      will get 401/403. Export API_KEY first, or run the server with ALLOW_NO_AUTH=1" >&2
  echo "      on loopback only." >&2
fi
echo "== /v1/models =="
curl -fsS "${AUTH[@]}" "$BASE/v1/models"; echo
echo "== /v1/chat/completions =="
curl -fsS "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"mimo-pro","messages":[{"role":"user","content":"Say TP4_OK"}],"max_tokens":16,"temperature":0}'
echo