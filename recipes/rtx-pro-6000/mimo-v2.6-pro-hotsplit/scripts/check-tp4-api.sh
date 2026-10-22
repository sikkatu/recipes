#!/usr/bin/env bash
# API smoke test for the TP4 server. Sends the API key when one is set; without a key the
# requests are only valid for an ALLOW_NO_AUTH loopback server (a protected API answers 401).
# usage: check-tp4-api.sh [base-url]   (default http://127.0.0.1:30007)
set -Eeuo pipefail
BASE="${1:-${BASE:-http://127.0.0.1:30007}}"
# the launcher persists its auto-generated key in live/api_key -- use it when the shell has none
if [ -z "${API_KEY:-}" ]; then
  for _kf in "live/api_key" "$PWD/live/api_key" "$(dirname "$0")/../live/api_key"; do
    if [ -f "$_kf" ]; then
      API_KEY="$(head -n1 "$_kf" | tr -d '[:space:]')"
      break
    fi
  done
fi
AUTH=()
if [ -n "${API_KEY:-}" ]; then
  AUTH=(-H "Authorization: Bearer $API_KEY")
else
  echo "NOTE: no API key found (shell, tp4.env or live/api_key). If the server requires" >&2
  echo "      auth (the default), these requests will get 401/403." >&2
  echo "      on loopback only." >&2
fi
echo "== /v1/models =="
curl -fsS "${AUTH[@]}" "$BASE/v1/models"; echo
echo "== /v1/chat/completions =="
curl -fsS "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"mimo-pro","messages":[{"role":"user","content":"Say TP4_OK"}],"max_tokens":64,"temperature":0}'
echo