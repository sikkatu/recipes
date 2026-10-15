#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${BASE:-http://127.0.0.1:30007}"
curl -fsS "$BASE/v1/models"; echo
curl -fsS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"mimo-pro","messages":[{"role":"user","content":"Say TP4_OK"}],"max_tokens":16,"temperature":0}'
echo
