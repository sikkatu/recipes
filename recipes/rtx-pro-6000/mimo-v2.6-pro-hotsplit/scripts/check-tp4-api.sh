#!/usr/bin/env bash
# API smoke test for the TP4 server. Sends the API key when one is set; without a key the
# requests are only valid for an ALLOW_NO_AUTH loopback server (a protected API answers 401).
# usage: check-tp4-api.sh [base-url]   (default http://127.0.0.1:30007)
set -Eeuo pipefail
# Round 65: send the API key on a stdin pipe (printf is a builtin -- the key is
# never visible in `ps` and no key file is needed). Without a key the request
# goes out unauthenticated (the documented ALLOW_NO_AUTH loopback mode).
kvp_curl(){
  if [ -n "${API_KEY:-}" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" | curl -K - "$@"
  else
    curl "$@"
  fi
}
BASE="${1:-${BASE:-http://127.0.0.1:30007}}"
# the launcher persists its auto-generated key in live/api_key -- use it when the shell has none
if [ -z "${API_KEY:-}" ]; then
  # Canonical shared key first (round 56/58), then the local mirror.
  for _kf in "${KVP_API_KEY_FILE:-$HOME/.config/kvp/api_key}" "live/api_key" "$PWD/live/api_key" "$(dirname "$0")/../live/api_key"; do
    if [ -f "$_kf" ]; then
      { IFS= read -r API_KEY || true; } < "$_kf"
      API_KEY="${API_KEY//[$'\t\r\n ']/}"
      break
    fi
  done
fi
AUTH=()
if [ -n "${API_KEY:-}" ]; then
  AUTH=()   # round 65: kvp_curl sends the key on a pipe -- nothing to pass here
else
  echo "NOTE: no API key found (shell, tp4.env or live/api_key). If the server requires" >&2
  echo "      auth (the default), these requests will get 401/403." >&2
  echo "      on loopback only." >&2
fi
echo "== /v1/models =="
kvp_curl -fsS "$BASE/v1/models"; echo
echo "== /v1/chat/completions =="
# Review fix (round 22): the model name must follow SERVED_MODEL_NAME -- a
# hardcoded "mimo-pro" gets 404 as soon as the launcher serves another name.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-pro}"
kvp_curl -fsS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say TP4_OK\"}],\"max_tokens\":64,\"temperature\":0}"
echo