#!/usr/bin/env bash
# MiMo-V2.6-Pro on 4x RTX PRO 6000: vLLM TP4 + UVA expert offload + hotsplit (static or dynamic hot experts).
# AUTHORED, NOT RUN ON TARGET HARDWARE. Read ../README.md first; run with HOTSPLIT_DRYRUN=1 once.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }

# --- security contract (SECURITY.md): no wide bind, no disabled auth -----------------------------
# BIND is where the SERVER listens (host networking: -p would be a no-op, so the bind is the only
# control). Loopback by default; API_KEY is required unless you explicitly opt out.
BIND="${BIND:-127.0.0.1}"
API_KEY="${API_KEY:-}"
ALLOW_NO_AUTH="${ALLOW_NO_AUTH:-0}"
# Order matters: a non-loopback bind ALWAYS requires a key, even with ALLOW_NO_AUTH=1.
# ALLOW_NO_AUTH=1 only permits an unauthenticated loopback listener (closed debug loop).
if [ "$BIND" != "127.0.0.1" ] && [ -z "$API_KEY" ]; then
  die "refusing a non-loopback bind ($BIND) without an API key -- that is a public unauthenticated API.
A key is mandatory for any non-loopback bind even when ALLOW_NO_AUTH=1."
fi
if [ -z "$API_KEY" ]; then
  if [ "$ALLOW_NO_AUTH" = "1" ] && [ "$BIND" = "127.0.0.1" ]; then
    echo "WARNING: running WITHOUT auth on loopback only (ALLOW_NO_AUTH=1) -- closed debug loop."
  else
    die "API_KEY is required (the API has no auth otherwise). Export a key (e.g. API_KEY=\$(openssl rand -hex 24)).
ALLOW_NO_AUTH=1 is accepted ONLY together with BIND=127.0.0.1 (loopback debug)."
  fi
fi

# --- contract: image digests, always (no :latest, no floating tags) -------------------------------
# The model patches below are built against a specific vLLM image, so the image MUST be pinned.
IMAGE="${IMAGE:-}"
case "$IMAGE" in
  *@sha256:*) ;;
  *) die "IMAGE must be pinned by digest (@sha256:...), got: '${IMAGE:-<empty>}'. Floating tags are not
reproducible and the mounted model patches only match one vLLM build. Get the digest first:
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
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$(cd "$HERE/.." && pwd)"
MODEL_ID="${MODEL_ID:-XiaomiMiMo/MiMo-V2.6-Pro-MOPD}"
MODEL_DIR="${MODEL_DIR:-}"
CONTAINER="${CONTAINER:-mimo-pro-tp4-hotsplit}"
SITE="${SITE:-}"   # empty = auto-detect inside the image; set explicitly to override
PATCH_MODEL_FILES="${PATCH_MODEL_FILES:-1}"   # 0 = keep the image's own mimo_v2.py; hook via sitecustomize
PORT="${PORT:-30007}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-96}"        # per rank; with hotsplit it only decides the initial placement
GPU_UTIL="${GPU_UTIL:-0.90}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
LIVE_DIR="${LIVE_DIR:-$PWD/live}"
# hotsplit knobs (documented in tp4.env.example)
HOTSPLIT_COUNTS="${HOTSPLIT_COUNTS:-/w/expert_hist_mix.json}"
HOTSPLIT_WEIGHTS="${HOTSPLIT_WEIGHTS:-decode=1,prefill=0}"
HOTSPLIT_HOT_GIB="${HOTSPLIT_HOT_GIB:-}"      # per rank; empty = same GPU-memory bytes as the stock placement
HOTSPLIT_DRYRUN="${HOTSPLIT_DRYRUN:-0}"
HOTSPLIT_DYNAMIC="${HOTSPLIT_DYNAMIC:-1}"
HOTSPLIT_ALLOW_STATIC_FALLBACK="${HOTSPLIT_ALLOW_STATIC_FALLBACK:-0}"
HOTSPLIT_HOST_FLOOR_GIB="${HOTSPLIT_HOST_FLOOR_GIB:-128}"
HOTSPLIT_NUMA="${HOTSPLIT_NUMA:-1}"
HOTSPLIT_NUMA_POLICY="${HOTSPLIT_NUMA_POLICY:-preferred}"

command -v docker >/dev/null || die "Docker is missing."
docker info >/dev/null 2>&1 || die "Docker daemon unavailable or permission denied."
command -v nvidia-smi >/dev/null || die "nvidia-smi missing."
command -v curl >/dev/null || die "curl missing."
command -v sha256sum >/dev/null || die "sha256sum missing."
N="$(nvidia-smi -L | wc -l | tr -d ' ')"
[ "$N" = 4 ] || die "Expected 4 GPUs, found $N."
docker container inspect "$CONTAINER" >/dev/null 2>&1 && die "Container already exists: $CONTAINER"
mkdir -p "$HF_CACHE" "$LIVE_DIR"

# --- contract: verify every mounted patch file against the pinned manifest before launch ----------
# PATCH_MODEL_FILES replaces vllm/.../mimo_v2.py wholesale, so a drifted patch silently changes the
# model. Refuse to launch unless the files on disk match patches/MANIFEST.sha256 exactly.
( cd "$RECIPE/patches" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1 ) \
  || die "patches/ do not match patches/MANIFEST.sha256. The model-replacement patches must be
byte-identical to the reviewed set; regenerate the manifest only after a deliberate change:
  ( cd patches && sha256sum mimo_v2_hotsplit.py mimo_v2_omni.py hotsplit.py hotsplit_plan.py numa_bind.py sitecustomize.py expert_hist_mix.json > MANIFEST.sha256 )"
echo "patch manifest verified against patches/MANIFEST.sha256"

for f in "$RECIPE/patches/hotsplit.py" "$RECIPE/patches/hotsplit_plan.py" "$RECIPE/patches/numa_bind.py" \
         "$RECIPE/patches/sitecustomize.py" "$RECIPE/patches/expert_hist_mix.json" \
         "$RECIPE/patches/mimo_v2_hotsplit.py" "$RECIPE/patches/mimo_v2_omni.py"; do
  [ -f "$f" ] || die "missing file: $f"
done

MODEL_ARG="$MODEL_ID"
MOUNTS=(-v "$HF_CACHE:/root/.cache/huggingface")
if [ -n "$MODEL_DIR" ]; then
  [ -d "$MODEL_DIR" ] || die "MODEL_DIR not found: $MODEL_DIR"
  MODEL_ARG=/model
  MOUNTS+=(-v "$MODEL_DIR:/model:ro")
fi
MOUNTS+=(
  -v "$RECIPE/patches/hotsplit.py:/w/hotsplit.py:ro"
  -v "$RECIPE/patches/hotsplit_plan.py:/w/hotsplit_plan.py:ro"
  -v "$RECIPE/patches/numa_bind.py:/w/numa_bind.py:ro"
  -v "$RECIPE/patches/sitecustomize.py:/w/sitecustomize.py:ro"
  -v "$RECIPE/patches/expert_hist_mix.json:/w/expert_hist_mix.json:ro"
  -v "$LIVE_DIR:/live"
)
BOOT_HOOK=1
if [ "$PATCH_MODEL_FILES" = 1 ]; then
  # --- patch-compatibility gate (audit: PATCH_MODEL_FILES replaces model files wholesale) ---
  # 1. Auto-detect the site-packages path inside the image (SITE= overrides).
  # 2. Verify the image's own mimo_v2.py / mimo_v2_omni.py still carry the API our patches
  #    override (load_weights, process_weights_after_loading, the MoE apply path) before we
  #    mount over them. A mismatch means the patches were written for a different vLLM build.
  # 3. Log the originals' SHA-256 for provenance (compare with a known-good list in the log).
  DETECT_CMD='import importlib.util,sys; s=importlib.util.find_spec("vllm"); print(s.submodule_search_locations[0])'
  if [ -z "$SITE" ]; then
    SITE="$(docker run --rm "$IMAGE" python -c "$DETECT_CMD" 2>/dev/null | tr -d '
')" || die "failed to detect vLLM site-packages inside the image"
    [ -n "$SITE" ] || die "empty vLLM site-packages path detected; set SITE= explicitly"
  fi
  echo "SITE (auto-detected or overridden): $SITE"
  # compatibility probe: the image's own files must expose the API our patches override
  PROBE_CMD='import importlib.util,sys
ok=True
for mod,name in [("vllm.model_executor.models.mimo_v2","mimo_v2"),("vllm.model_executor.models.mimo_v2_omni","mimo_v2_omni")]:
    try:
        m=importlib.util.find_spec(mod)
        if m is None:
            print(f"MISSING {name}"); ok=False; continue
        print(f"FOUND {name}: {m.origin}")
        src=open(m.origin).read()
        for fn in ["load_weights","process_weights_after_loading"]:
            if fn in src:
                print(f"  API {name}.{fn}: present")
            else:
                print(f"  API {name}.{fn}: ABSENT")
                ok=False
    except Exception as e:
        print(f"ERROR {name}: {e!r}"); ok=False
sys.exit(0 if ok else 3)'
  echo "Probing the image's own model files for patch compatibility..."
  if ! docker run --rm "$IMAGE" python -c "$PROBE_CMD"; then
    die "the image's vLLM model files do not match what patches/mimo_v2_hotsplit.py expects.
The patches replace load_weights / process_weights_after_loading wholesale; if those methods are
missing or renamed in this build the mount will break the model. Either use PATCH_MODEL_FILES=0
(hook via sitecustomize) or rebase the patches onto this exact vLLM build."
  fi
  # provenance: log the originals' hashes before we shadow them
  HASH_CMD='import hashlib,importlib.util
for mod in ["vllm.model_executor.models.mimo_v2","vllm.model_executor.models.mimo_v2_omni"]:
    s=importlib.util.find_spec(mod)
    if s and s.origin:
        h=hashlib.sha256(open(s.origin,"rb").read()).hexdigest()
        print(f"{h}  {s.origin}")'
  echo "Original module SHA-256 (for provenance; compare with your known-good list):"
  docker run --rm "$IMAGE" python -c "$HASH_CMD" || true
  MOUNTS+=(
    -v "$RECIPE/patches/mimo_v2_hotsplit.py:$SITE/vllm/model_executor/models/mimo_v2.py:ro"
    -v "$RECIPE/patches/mimo_v2_omni.py:$SITE/vllm/model_executor/models/mimo_v2_omni.py:ro"
  )
  BOOT_HOOK=0   # the replaced mimo_v2.py already calls hotsplit.apply_hotsplit
fi

ENVS=(
  -e HF_HOME=/root/.cache/huggingface
  -e PYTHONPATH=/w
  -e VLLM_USE_DEEP_GEMM=0
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
  -e "VLLM_API_KEY=$API_KEY"
  -e "HOTSPLIT_COUNTS=$HOTSPLIT_COUNTS"
  -e "HOTSPLIT_WEIGHTS=$HOTSPLIT_WEIGHTS"
  -e "HOTSPLIT_DRYRUN=$HOTSPLIT_DRYRUN"
  -e "HOTSPLIT_DYNAMIC=$HOTSPLIT_DYNAMIC"
  -e "HOTSPLIT_ALLOW_STATIC_FALLBACK=$HOTSPLIT_ALLOW_STATIC_FALLBACK"
  -e "HOTSPLIT_HOST_FLOOR_GIB=$HOTSPLIT_HOST_FLOOR_GIB"
  -e "HOTSPLIT_NUMA=$HOTSPLIT_NUMA"
  -e "HOTSPLIT_NUMA_POLICY=$HOTSPLIT_NUMA_POLICY"
  -e "HOTSPLIT_BOOT_HOOK=$BOOT_HOOK"
  -e HOTSPLIT_LIVE_COUNTS=/live/counts.json
  -e HOTSPLIT_LIVE_SECS=600
)
[ -n "$HOTSPLIT_HOT_GIB" ] && ENVS+=(-e "HOTSPLIT_HOT_GIB=$HOTSPLIT_HOT_GIB")
for v in HOTSPLIT_DYN_STEPS HOTSPLIT_DYN_MAX_SWAPS HOTSPLIT_DYN_DECAY HOTSPLIT_DYN_MARGIN \
         HOTSPLIT_DYN_PRIOR_TOKENS HOTSPLIT_DYN_LOG HOTSPLIT_TOPK HOTSPLIT_LIVE_DECODE_MAX; do
  [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
done

# --- model arguments: arrays, never re-parsed shell strings --------------------------------------
MODEL_ARGS=(--model "$MODEL_ARG")
if [ -z "$MODEL_DIR" ]; then
  # remote fetch: pin the checkpoint commit SHA (validated above)
  MODEL_ARGS+=(--revision "$MODEL_REVISION")
fi
docker run -d --name "$CONTAINER" --restart no \
  --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" \
  "${MODEL_ARGS[@]}" \
  --served-model-name mimo-pro \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce \
  --offload-backend uva \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --moe-backend marlin \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --generation-config vllm \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --host "$BIND" --port "$PORT"

WAIT_TRIES="${WAIT_TRIES:-180}"
WAIT_SLEEP="${WAIT_SLEEP:-10}"   # readiness polls, 10s apart (tests override this)
echo "Waiting up to $((WAIT_TRIES * 10)) seconds for API readiness (docker logs -f $CONTAINER)..."
for _ in $(seq 1 "$WAIT_TRIES"); do
  if curl -fsS -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/tmp/mimo-models.json; then
    cat /tmp/mimo-models.json; echo
    curl -fsS -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
      "http://127.0.0.1:$PORT/v1/chat/completions" \
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