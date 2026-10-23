#!/usr/bin/env bash
# MiMo-V2.6-Pro on 4x RTX PRO 6000: vLLM TP4 + UVA expert offload baseline (no hot/cold split).
# VERIFIED on target hardware (2026-10-22): weights load, API answers, generation coherent. Read ../README.md first.
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
    head -c 24 /dev/urandom | od -An -tx1 | tr -d '[:space:]'; echo
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
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"        # auto = bf16; fp8 halves KV memory (use for long context)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"   # raise to 8192 for long-context prefill
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-128}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MOE_BACKEND="${MOE_BACKEND:-humming}"       # OBSERVED on target hardware: triton rejects MoEActivation.SILU for MiMo;
#                # marlin repacks weights into VRAM and OOMs with UVA offload. humming supports SILU and does not repack.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$(cd "$HERE/.." && pwd)"
# --- fixed model files (A/B-confirmed on target hardware, 2026-10-22) -----------------------------
# The image's own vllm/model_executor/models/mimo_v2.py load_weights silently drops fused
# qkv_proj weight+weight_scale_inv pairs (they can straddle two AutoWeightsLoader calls), and
# the model then emits training-corpus fragments instead of text. patches/mimo_v2_hotsplit.py
# carries a loader that keeps those pairs together. A/B with identical weights and flags:
#   PATCH_MODEL_FILES=1 -> coherent text (" Paris. In the early Middle Ages...")
#   PATCH_MODEL_FILES=0 -> garbage ("            0     0,   0,  00 ")
# Default 1 = working output. Set 0 only to reproduce the upstream bug on purpose.
# The patch's trailing hotsplit hook is gated on HOTSPLIT_COUNTS in the container env; this
# launcher never forwards that variable, so the hook stays OFF and only the loader is fixed.
PATCH_MODEL_FILES="${PATCH_MODEL_FILES:-1}"
SITE="${SITE:-}"   # empty = auto-detect inside the image; set explicitly to override
command -v docker >/dev/null || die "Docker is missing."
docker info >/dev/null 2>&1 || die "Docker daemon unavailable or permission denied."
command -v nvidia-smi >/dev/null || die "nvidia-smi missing."
command -v curl >/dev/null || die "curl missing."
command -v sha256sum >/dev/null || die "sha256sum missing."
N="$(nvidia-smi -L | wc -l | tr -d ' ')"
[ "$N" = 4 ] || die "Expected 4 GPUs, found $N."
docker container inspect "$CONTAINER" >/dev/null 2>&1 && die "Container already exists: $CONTAINER"
# --- host RAM preflight: pinned UVA memory is TP x CPU_OFFLOAD_GB and is unswappable ------
# With 4 ranks the host must hold ALL of it plus room to stream the checkpoint shards.
# Warn (not fail: the kernel page cache can absorb some slack) when the budget looks too tight.
TOTAL_RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
AVAIL_RAM_KB="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
case "$CPU_OFFLOAD_GB" in
  ''|*[!0-9]*) echo "ERROR: CPU_OFFLOAD_GB must be a whole number of GiB, got '${CPU_OFFLOAD_GB}'" >&2; exit 1 ;;
esac
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
# --- patch gate: mount the fixed model files over the image's own ------------------------------
# Same contract as run-tp4-hotsplit.sh: verify the mounted files against the pinned manifest,
# auto-detect site-packages INSIDE the image, probe that the image's own files expose the API
# the patch replaces (load_weights), and log the originals' SHA-256 for provenance.
if [ "$PATCH_MODEL_FILES" = 1 ]; then
  for f in "$RECIPE/patches/mimo_v2_hotsplit.py" "$RECIPE/patches/mimo_v2_omni.py" \
           "$RECIPE/patches/MANIFEST.sha256"; do
    [ -f "$f" ] || die "missing file: $f"
  done
  ( cd "$RECIPE/patches" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1 ) \
    || die "patches/ do not match patches/MANIFEST.sha256. The model-replacement patches must be
byte-identical to the reviewed set; regenerate the manifest only after a deliberate change:
  ( cd patches && sha256sum mimo_v2_hotsplit.py mimo_v2_omni.py hotsplit.py hotsplit_plan.py numa_bind.py sitecustomize.py expert_hist_mix.json > MANIFEST.sha256 )"
  echo "patch manifest verified against patches/MANIFEST.sha256"
  # find_spec("vllm") returns the PACKAGE dir (.../dist-packages/vllm). The mount targets
  # $SITE/vllm/model_executor/models/, so SITE must be the site-packages ROOT (the parent).
  # The path only exists INSIDE the container -- validate it there, never with [ -d ] on the host.
  DETECT_CMD='import importlib.util,os,sys
s=importlib.util.find_spec("vllm")
if s is None:
    sys.exit("vllm package not found inside the image")
locs=list(s.submodule_search_locations or [])
pkg = locs[0] if locs else os.path.dirname(s.origin)
site = os.path.dirname(pkg)   # .../site-packages/vllm -> .../site-packages
if not os.path.isdir(os.path.join(site, "vllm")):
    sys.exit("site-packages root has no vllm/: " + site)
print(site)'
  PYBIN=python3
  docker run --rm --entrypoint python3 "$IMAGE" -c 'import sys; print(sys.version)' >/dev/null 2>&1 || PYBIN=python
  if [ -z "$SITE" ]; then
    SITE_OUT="$(docker run --rm --entrypoint $PYBIN "$IMAGE" -c "$DETECT_CMD" 2>&1)" || true
    SITE="$(printf '%s\n' "$SITE_OUT" | tail -n1 | tr -d '\r' | sed 's/[[:space:]]*$//')"
    case "$SITE" in
      /*/site-packages|/*/dist-packages) : ;;
      *) die "failed to detect vLLM site-packages inside the image (entrypoint=$PYBIN).
Probe output:
$SITE_OUT
Expected a site-packages root such as /usr/local/lib/python3.12/dist-packages.
Set SITE= explicitly to skip the probe." ;;
    esac
  fi
  echo "SITE (auto-detected or overridden): $SITE"
  # compatibility probe: the image's own files must expose load_weights (what the patch replaces)
  PROBE_CMD='import importlib.util,sys
ok=True
for mod,name in [("vllm.model_executor.models.mimo_v2","mimo_v2"),("vllm.model_executor.models.mimo_v2_omni","mimo_v2_omni")]:
    try:
        m=importlib.util.find_spec(mod)
        if m is None:
            print("MISSING %s" % name); ok=False; continue
        print("FOUND %s: %s" % (name, m.origin))
        src=open(m.origin).read()
        if "def load_weights" in src:
            print("  API %s.load_weights: present" % name)
        else:
            print("  API %s.load_weights: ABSENT" % name); ok=False
    except Exception as e:
        print("ERROR %s: %r" % (name, e)); ok=False
sys.exit(0 if ok else 3)'
  echo "Probing the image's own model files for patch compatibility..."
  if ! docker run --rm --entrypoint $PYBIN "$IMAGE" -c "$PROBE_CMD"; then
    die "the image's vLLM model files do not match what patches/mimo_v2_hotsplit.py expects.
The patch replaces load_weights wholesale; if that method is missing or renamed in this
build the mount will break the model. Set PATCH_MODEL_FILES=0 to use the image's own files
(known to emit garbage on target hardware) or rebase the patches onto this exact vLLM build."
  fi
  HASH_CMD='import hashlib,importlib.util
for mod in ["vllm.model_executor.models.mimo_v2","vllm.model_executor.models.mimo_v2_omni"]:
    s=importlib.util.find_spec(mod)
    if s and s.origin:
        h=hashlib.sha256(open(s.origin,"rb").read()).hexdigest()
        print(f"{h}  {s.origin}")'
  echo "Original module SHA-256 (for provenance; compare with your known-good list):"
  docker run --rm --entrypoint $PYBIN "$IMAGE" -c "$HASH_CMD" || true
  MOUNTS+=(
    -v "$RECIPE/patches/mimo_v2_hotsplit.py:$SITE/vllm/model_executor/models/mimo_v2.py:ro"
    -v "$RECIPE/patches/mimo_v2_omni.py:$SITE/vllm/model_executor/models/mimo_v2_omni.py:ro"
  )
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
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-prefix-caching \
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
echo "Waiting up to $((WAIT_TRIES * 10)) seconds for API readiness. Progress prints every ${WAIT_SLEEP}s (new log lines only):"
LOG_SHOWN=0
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
  # progress line every poll: print only NEW log lines (a repeated identical line every
  # 10s reads as a hang); when nothing new, say so instead of reprinting the same text.
  ELAPSED=$(( $(date +%s) - START_TS ))
  LOG_ALL="$(docker logs "$CONTAINER" 2>&1)"
  LOG_TOTAL="$(printf '%s\n' "$LOG_ALL" | wc -l)"
  NEW_COUNT=0
  if [ "$LOG_TOTAL" -gt "${LOG_SHOWN:-0}" ]; then
    NEW_COUNT=$(( LOG_TOTAL - LOG_SHOWN ))
    printf '%s\n' "$LOG_ALL" | tail -n "$NEW_COUNT" | grep -E 'Loading|loaded|weights|safetensors|offload|Ready|Starting|server|UVA|compile|graph|CUDA|ERROR|WARNING' | cut -c1-120 || true
    LOG_SHOWN="$LOG_TOTAL"
  fi
  if [ "$NEW_COUNT" -eq 0 ]; then
    LASTLOG="$(printf '%s\n' "$LOG_ALL" | grep -E 'Loading|loaded|weights|safetensors|offload|Ready|Starting|server|UVA|compile|graph|CUDA' | tail -1 | cut -c1-90 || true)"
    printf '[%3ds] container=%s  still working (no new lines yet; last: %s)\n' "$ELAPSED" "${RUN_STATE%% *}" "${LASTLOG:-starting...}"
  else
    printf '[%3ds] container=%s  %d new log line(s)\n' "$ELAPSED" "${RUN_STATE%% *}" "$NEW_COUNT"
  fi
  if curl -fsS --max-time 3 -H "Authorization: Bearer $API_KEY" "http://$URLHOST:$PORT/v1/models" > "$MODELS_JSON" 2>/dev/null; then
    cat "$MODELS_JSON"; echo
    SMOKE_JSON="$(curl -fsS -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
      "http://$URLHOST:$PORT/v1/chat/completions" \
      -d '{"model":"mimo-pro","messages":[{"role":"user","content":"Reply exactly TP4_OK"}],"max_tokens":64,"temperature":0}')"
    echo "$SMOKE_JSON"
    # the answer must actually contain TP4_OK: HTTP 200 + a coherent-looking body is not
    # proof of correct weights (the stock loader produced garbage that looked plausible)
    if printf '%s' "$SMOKE_JSON" | grep -q '"content":"[^"]*TP4_OK'; then
      echo "smoke test: answer contains TP4_OK -- model output verified"
    else
      echo "ERROR: smoke answer does NOT contain TP4_OK -- weights may be loaded wrong" >&2
      echo "       (compare with a run using PATCH_MODEL_FILES=1; see README Known limits)" >&2
    fi
    echo "Baseline ready (no hot/cold split in this launcher). Verify the API with: bash $HERE/check-tp4-api.sh"
    exit 0
  fi
  sleep "$WAIT_SLEEP"
done
echo "Startup timed out; last logs:" >&2
docker logs --tail 250 "$CONTAINER" >&2 || true
exit 1
