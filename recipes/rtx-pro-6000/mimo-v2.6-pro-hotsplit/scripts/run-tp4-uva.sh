#!/usr/bin/env bash
# MiMo-V2.6-Pro on 4x RTX PRO 6000: vLLM TP4 + UVA expert offload baseline (no hot/cold split).
# AUTHORED, NOT RUN ON TARGET HARDWARE. Read ../README.md first.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }

# --- security contract (SECURITY.md): no wide bind, no disabled auth -----------------------------
# BIND is where the SERVER listens (host networking: -p would be a no-op, so the bind is the
# only control). Loopback by default; API_KEY is required unless you explicitly opt out.
BIND="${BIND:-127.0.0.1}"
BIND="${BIND#\[}"; BIND="${BIND%\]}"   # accept [::1]; vLLM --host wants the bare address
API_KEY="${API_KEY:-}"
ALLOW_NO_AUTH="${ALLOW_NO_AUTH:-0}"
# --- API key handling ---------------------------------------------------------------------------
# The key is persisted in live/api_key (chmod 600, gitignored) and reused across restarts.
# On a LOOPBACK bind with no key one is auto-generated -- the user never exports anything.
# On any OTHER bind an explicit key is MANDATORY and never auto-generated (fail-safe: a
# server on a routable address is always authenticated). ALLOW_NO_AUTH=1 is accepted ONLY
# together with a proven loopback bind (127.0.0.1 or ::1).
LIVE_DIR="${LIVE_DIR:-$PWD/live}"
KEY_FILE="$LIVE_DIR/api_key"
mkdir -p "$LIVE_DIR"
if [ -z "$API_KEY" ] && [ -f "$KEY_FILE" ]; then
  API_KEY="$(head -n1 "$KEY_FILE" | tr -d '[:space:]')"
fi
gen_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import secrets; print(secrets.token_hex(24))'
  else
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \\n'; echo
  fi
}
# Order matters: a non-loopback bind ALWAYS requires a key, even with ALLOW_NO_AUTH=1.
# ALLOW_NO_AUTH=1 only permits an unauthenticated loopback listener (closed debug loop).
# Loopback is strictly 127.0.0.1 or ::1 (optionally bracketed). Everything else -- including
# `localhost` (resolves via getaddrinfo, we cannot reason about the result), wildcard binds
# (0.0.0.0, ::, *), other IPv6 and any malformed or empty value -- is treated as NON-loopback
# and needs an API key. Fail-safe: an unrecognized address never gets the no-auth bypass.
if [ -z "$BIND" ]; then
  die "BIND is empty. Use a literal IP: 127.0.0.1 (loopback), ::1 (IPv6 loopback) or a routable address."
fi
LOOPBACK=0
case "$BIND" in
  127.0.0.1|\[::1\]|::1) LOOPBACK=1 ;;
esac
if [ "$LOOPBACK" = "0" ] && [ -z "$API_KEY" ]; then
  die "refusing a non-loopback bind ($BIND) without an API key -- that is a public unauthenticated API.
A key is mandatory for any non-loopback bind even when ALLOW_NO_AUTH=1."
fi
if [ -z "$API_KEY" ]; then
  if [ "$ALLOW_NO_AUTH" = "1" ] && [ "$LOOPBACK" = "1" ]; then
    echo "WARNING: running WITHOUT auth on loopback only (ALLOW_NO_AUTH=1) -- closed debug loop."
  else
    API_KEY="$(gen_key)"
    case "$API_KEY" in
      *[!0-9a-f]*|'') die "failed to generate an API key (openssl/python3/urandom all unavailable)" ;;
    esac
    [ "${#API_KEY}" -ge 32 ] || die "generated API key is too short (${#API_KEY} chars)"
    umask 077
    printf '%s\n' "$API_KEY" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "API key auto-generated for the loopback server and saved to $KEY_FILE"
    echo "(not printed here; check-tp4-api.sh and manage.sh read it automatically)."
  fi
fi
if [ -n "$API_KEY" ] && [ ! -f "$KEY_FILE" ]; then
  umask 077
  printf '%s\n' "$API_KEY" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

# --- contract: image digests, always (no :latest, no floating tags) -------------------------------
IMAGE="${IMAGE:-}"
case "$IMAGE" in
  *@sha256:*)
    # the digest must be exactly 64 hex chars; a malformed one silently resolves to a
    # different image or fails later in a confusing way
    if ! printf '%s' "$IMAGE" | grep -Eq '@sha256:[0-9a-f]{64}$'; then
      die "IMAGE digest is malformed: '$IMAGE'
Expected @sha256:<exactly 64 hex chars>. Get the real digest with:
  docker buildx imagetools inspect <repo>:<tag> | grep Digest"
    fi ;;
  *) die "IMAGE must be pinned by digest (@sha256:...), got: '${IMAGE:-<empty>}'.
Floating tags are not reproducible. Get the digest first:
  docker buildx imagetools inspect <repo>:<tag> | grep Digest
then run: IMAGE=<repo>:<tag>@sha256:<digest> $0" ;;
esac

# --- contract: model revisions are commit SHAs when the model is fetched --------------------------
MODEL_REVISION="${MODEL_REVISION:-}"
if [ -n "${MODEL_DIR:-}" ]; then
  : # local checkpoint dir: the revision is whatever is on disk; record it yourself
elif [ -z "$MODEL_REVISION" ]; then
  die "MODEL_REVISION is required when the model is fetched by ID (trust-remote-code runs whatever is
on the ref). Pin the checkpoint commit SHA, or use MODEL_DIR with a local copy you vetted."
elif ! printf '%s' "$MODEL_REVISION" | grep -Eq '^[0-9a-f]{40}$'; then
  die "MODEL_REVISION must be a full 40-hex commit SHA (got '$MODEL_REVISION').
Branch names like 'main' are NOT reproducible -- the model is fetched by id here.
Get the SHA with:
  curl -fsSL https://huggingface.co/api/models/<MODEL_ID> | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"sha\"])'"
fi

MODEL_ID="${MODEL_ID:-XiaomiMiMo/MiMo-V2.6-Pro-MOPD}"
MODEL_DIR="${MODEL_DIR:-}"
CONTAINER="${CONTAINER:-mimo-pro-tp4-uva}"
PORT="${PORT:-30007}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-128}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MOE_BACKEND="${MOE_BACKEND:-humming}"       # OBSERVED on target hardware: triton rejects MoEActivation.SILU for MiMo;
#                # marlin repacks weights into VRAM and OOMs with UVA offload. humming supports SILU and does not repack.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HOTSPLIT_COUNTS="${HOTSPLIT_COUNTS:-/w/expert_hist_mix.json}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v docker >/dev/null || die "Docker is missing."
docker info >/dev/null 2>&1 || die "Docker daemon unavailable or permission denied."
command -v nvidia-smi >/dev/null || die "nvidia-smi missing."
command -v curl >/dev/null || die "curl missing."
N="$(nvidia-smi -L | wc -l | tr -d ' ')"
[ "$N" = 4 ] || die "Expected 4 GPUs, found $N."
docker container inspect "$CONTAINER" >/dev/null 2>&1 && die "Container already exists: $CONTAINER"
# --- host RAM preflight: pinned UVA memory is TP x CPU_OFFLOAD_GB and is unswappable ------
# With 4 ranks the host must hold ALL of it plus room to stream the checkpoint shards.
# Warn (not fail: the kernel page cache can absorb some slack) when the budget looks too tight.
TOTAL_RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
AVAIL_RAM_KB="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
PINNED_GIB=$((CPU_OFFLOAD_GB * 4))
NEED_GIB=$((PINNED_GIB + 24))
AVAIL_GIB=$((AVAIL_RAM_KB / 1024 / 1024))
if [ "$TOTAL_RAM_KB" -gt 0 ] && [ "$AVAIL_GIB" -lt "$NEED_GIB" ]; then
  echo "WARNING: host RAM looks tight for UVA offload." >&2
  echo "  pinned (4 x CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB) = ${PINNED_GIB} GiB + ~24 GiB load buffer = ${NEED_GIB} GiB needed" >&2
  echo "  currently available: ${AVAIL_GIB} GiB. Loading a large checkpoint may hit the OOM killer." >&2
  echo "  Give the VM more RAM (e.g. 800 GiB for MiMo at CPU_OFFLOAD_GB=128) or lower CPU_OFFLOAD_GB." >&2
fi
mkdir -p "$HF_CACHE" "$LIVE_DIR"

MODEL_ARG="$MODEL_ID"
MOUNTS=(-v "$HF_CACHE:/root/.cache/huggingface")
if [ -n "$MODEL_DIR" ]; then
  [ -d "$MODEL_DIR" ] || die "MODEL_DIR not found: $MODEL_DIR"
  MODEL_ARG=/model
  MOUNTS+=(-v "$MODEL_DIR:/model:ro")
fi

# --- model arguments: arrays, never re-parsed shell strings --------------------------------------
MODEL_ARGS=(--model "$MODEL_ARG")
if [ -z "$MODEL_DIR" ]; then
  # remote fetch: pin the checkpoint commit SHA (validated above)
  MODEL_ARGS+=(--revision "$MODEL_REVISION")
fi
docker run -d --name "$CONTAINER" --restart no \
  --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  -e "VLLM_API_KEY=$API_KEY" \
  "${MOUNTS[@]}" "$IMAGE" \
  "${MODEL_ARGS[@]}" \
  --served-model-name mimo-pro \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce \
  --offload-backend uva \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --moe-backend "$MOE_BACKEND" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config vllm \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --host "$BIND" --port "$PORT"
# URL host for the readiness probes: bracket IPv6, probe wildcard binds via loopback
case "$BIND" in
  0.0.0.0) URLHOST=127.0.0.1 ;;
  ::)      URLHOST='[::1]' ;;
  *:*)     URLHOST="[$BIND]" ;;
  *)       URLHOST="$BIND" ;;
esac
MODELS_JSON="$(mktemp /tmp/tp4-models.XXXXXX.json)"
WAIT_TRIES="${WAIT_TRIES:-180}"
WAIT_SLEEP="${WAIT_SLEEP:-10}"   # readiness polls, 10s apart (tests override this)
echo "Waiting up to $((WAIT_TRIES * 10)) seconds for API readiness. Progress prints every ${WAIT_SLEEP}s:"
START_TS="$(date +%s)"
for _ in $(seq 1 "$WAIT_TRIES"); do
  # fail fast: if the container died, waiting cannot help -- show why and stop now
  RUN_STATE="$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$CONTAINER" 2>/dev/null || echo gone)"
  case "$RUN_STATE" in
    running*) : ;;
    *)
      echo "ERROR: container $CONTAINER is not running ($RUN_STATE) -- the server never started." >&2
      echo "Last logs:" >&2
      docker logs --tail 250 "$CONTAINER" >&2 || true
      exit 1 ;;
  esac
  # progress line every poll: tells the user the server is alive and what it is doing
  ELAPSED=$(( $(date +%s) - START_TS ))
  LASTLOG="$(docker logs --tail 400 "$CONTAINER" 2>&1 | grep -E 'Loading|loaded|weights|safetensors|offload|Ready|Starting|server|UVA|memory' | tail -1 | cut -c1-110 || true)"
  printf '[%3ds] container=%s  %s\n' "$ELAPSED" "${RUN_STATE%% *}" "${LASTLOG:-starting...}"
  if curl -fsS --max-time 3 -H "Authorization: Bearer $API_KEY" "http://$URLHOST:$PORT/v1/models" > "$MODELS_JSON" 2>/dev/null; then
    cat "$MODELS_JSON"; echo
    curl -fsS -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
      "http://$URLHOST:$PORT/v1/chat/completions" \
      -d '{"model":"mimo-pro","messages":[{"role":"user","content":"Reply exactly TP4_OK"}],"max_tokens":16,"temperature":0}'
    echo
    echo "Now run: $HERE/check-hotsplit-tp4.sh $CONTAINER"
    exit 0
  fi
  sleep "$WAIT_SLEEP"
done
echo "Startup timed out; last logs:" >&2
docker logs --tail 250 "$CONTAINER" >&2 || true
exit 1
