#!/usr/bin/env bash
# MiMo-V2.6-Pro on 4x RTX PRO 6000: vLLM TP4 + UVA expert offload baseline (no hot/cold split).
# VERIFIED on target hardware (2026-10-22): weights load, API answers, generation coherent. Read ../README.md first.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
# --- seed settings from ./tp4.env (optional) ------------------------------------------------------
# Same contract as manage.sh: values already in the environment WIN over the file -- the file
# is applied only for names that are not already exported ("only fill values not already
# exported"). Copy tp4.env.example to tp4.env and edit; no 'set -a' sourcing (that would lose
# the env-priority contract). The file is read before any validation so IMAGE / MODEL_REVISION /
# API_KEY set in it are honoured.
_TP4_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tp4.env"
if [ -f "$_TP4_ENV" ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      ''|'#'*) continue ;;                 # skip blanks and comments
    esac
    _line="${_line#export }"               # allow 'export X=1' without gluing it into the name
    _name="${_line%%=*}"
    [ "$_name" = "$_line" ] && continue     # not an assignment
    _name="$(printf '%s' "$_name" | tr -d ' ')"
    [ -z "$_name" ] && continue
    if [ -z "${!_name:-}" ]; then           # only fill values not already exported
      _val="${_line#*=}"
      _val="${_val%% #*}"                   # strip inline comment: everything after " # "
      _val="${_val%"${_val##*[![:space:]]}"}"   # strip trailing whitespace
      _val="${_val//\"/}"                  # strip surrounding double quotes if any
      case "$_val" in
        *'$HOME'*|*'${HOME}'*) _val="${_val//\$HOME/$HOME}"; _val="${_val//\$\{HOME\}/$HOME}" ;;
      esac
      export "$_name=$_val"
    fi
  done < "$_TP4_ENV"
  echo "settings seeded from $_TP4_ENV (exported environment still wins)"
fi

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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"   # 1M context by default (model max_position_embeddings=1048576)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"        # auto = the model default (bf16 for MiMo; the VERIFIED benchmarks ran on auto); fp8 halves KV memory (use for long context)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"   # long-context prefill (1M context default)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
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
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "NOTE: removing leftover container $CONTAINER from a previous run..."
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || die "could not remove $CONTAINER — remove it manually (docker rm -f $CONTAINER)"
fi
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
# --- persistent KV cache (kv-persistence; see github.com/sikkatu/kv-persistence) -----------------
# OFF by default (KV_PERSIST unset/0 changes nothing). When KV_PERSIST=1 the launcher
# attaches the external LMCache connector and additionally guarantees the KV
# cache identity namespace (see the LONG note below), passes
# --enable-prompt-tokens-details (needed for cached_tokens / acceptance runs)
# and pins --block-size so the kv-persistence identity gate can verify it.
# Every other argument, patch and the image stay identical.
#
# WHY THE NAMESPACE STEP (review fix): LMCache derives its cache identity
# (ObjectKey + L2 file name) from vLLM's `model_config.model` = whatever
# `--model` was passed. The plain recipe mounts MODEL_DIR at /model, so EVERY
# deployment would register the same LMCache model_name and different
# revisions/dtypes/geometries would share one L2 namespace. kv-persistence's
# own launcher (run-vllm-with-persist.sh) fixed this by mounting the model
# under a namespace-coded path derived from the verified cache identity. This
# block applies the SAME mechanism here:
#   1. prefer the real identity hash from a local kv-persistence checkout
#      (KVP_REPO_ROOT / ~/kv-persistence): `python3 -m kvpersist.cli model-name`.
#      The REAL engine arguments are passed through as KVP_* overrides so the
#      namespace follows the runtime values, not the shipped pins. KV_CACHE_DTYPE
#      is RESOLVED first (`auto` -> the model's actual dtype from config.json) so
#      an `auto` launch hashes the same as a `bfloat16` launch of that model
#      (review fix: `KV_CACHE_DTYPE=fp8` used to hash as the shipped bf16 pin --
#      a wrong-hit risk, not a safe miss);
#   2. fall back to a deterministic hash of the fields that actually change
#      KV compatibility (weights files, config.json, tokenizer files, chat
#      template, kv-cache dtype, max-model-len, block size, TP size) when
#      kv-persistence is not installed.
# The result is a path like /models/<namespace>; the engine then reports that
# path as model_config.model and incompatible deployments can never read each
# other's KV blocks (a safe cache miss -- the namespace covers every
# KV-affecting field, so any difference in one of them lands in a different
# namespace).
KV_PERSIST="${KV_PERSIST:-0}"
if [ "$KV_PERSIST" = "1" ]; then
  KV_PERSIST_HOST="${KV_PERSIST_HOST:-127.0.0.1}"
  KV_PERSIST_PORT="${KV_PERSIST_PORT:-5555}"
  KV_PERSIST_BLOCK_SIZE="${KV_PERSIST_BLOCK_SIZE:-16}"
  case "$KV_PERSIST_HOST:$KV_PERSIST_PORT" in
    *[^0-9a-zA-Z.\[\]:-]*) die "KV_PERSIST_HOST/PORT contain unexpected characters" ;;
  esac
  # Review fix (round 23): resolve the EFFECTIVE engine config through the
  # SAME shared code the namespace derivation uses (kvpersist.cli
  # effective-identity when a kv-persistence checkout is present, identical
  # inline rules otherwise) and PERSIST it to $LIVE_DIR/kvp-effective-identity.env.
  # The manual identity-gate run (reminder at the end of this block) sources
  # that file, so it compares the container args against the SAME values this
  # launch hashed the namespace from -- a non-default run (KV_CACHE_DTYPE=fp8,
  # MAX_MODEL_LEN=524288, TP!=4) verifies clean instead of being blocked by a
  # MISMATCH against the shipped identity.env pins.
  KVP_REPO="${KVP_REPO_ROOT:-$HOME/kv-persistence}"
  _EFF_ENV=""
  if [ -f "$KVP_REPO/implementation/kvpersist/cli.py" ]; then
    _EFF_ENV="$(PYTHONPATH="$KVP_REPO/implementation${PYTHONPATH:+:$PYTHONPATH}" \
      python3 -m kvpersist.cli effective-identity ${MODEL_DIR:+--model-dir "$MODEL_DIR"} 2>/dev/null || true)"
  fi
  if [ -z "$_EFF_ENV" ]; then
    # kv-persistence not importable here -- identical resolution rules.
  _EFF_ENV="$(python3 - "${MODEL_DIR:-}" <<'PYEFF'
import json, os, sys
def pick(*names, default=""):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default
kv_dtype = pick("KVP_KV_CACHE_DTYPE", "KV_CACHE_DTYPE", default="auto").lower()
model_dir = sys.argv[1] if len(sys.argv) > 1 else ""
if kv_dtype == "auto" and model_dir:
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    except Exception:
        cfg = {}
    for key in ("torch_dtype", "dtype"):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            kv_dtype = v.strip().lower()
            break
print("KVP_SERVED_MODEL_NAME=%s" % pick("KVP_SERVED_MODEL_NAME", "SERVED_MODEL_NAME", default="mimo-pro"))
print("KVP_KV_CACHE_DTYPE=%s" % kv_dtype)
print("KVP_MAX_MODEL_LEN=%s" % pick("KVP_MAX_MODEL_LEN", "MAX_MODEL_LEN", default="1048576"))
# Review fix (round 26): bare BLOCK_SIZE deliberately NOT read -- the knob
# was renamed to KV_PERSIST_BLOCK_SIZE to avoid clashes with unrelated
# environments (a stray BLOCK_SIZE=64 would desync the gate file from the
# engine arg).
print("KVP_BLOCK_SIZE=%s" % pick("KVP_BLOCK_SIZE", "KV_PERSIST_BLOCK_SIZE", default="16"))
print("KVP_TP_SIZE=%s" % pick("KVP_TP_SIZE", "TP_SIZE", default="4"))
PYEFF
)"
  fi
  [ -n "$_EFF_ENV" ] || die "cannot resolve the effective engine identity (kvpersist CLI and inline fallback both failed)"
  printf '%s\n' "$_EFF_ENV" > "$LIVE_DIR/kvp-effective-identity.env"
  echo "KV_PERSIST=1: effective engine identity saved to $LIVE_DIR/kvp-effective-identity.env"
  KV_EFFECTIVE_DTYPE="$(printf '%s\n' "$_EFF_ENV" | sed -n 's/^KVP_KV_CACHE_DTYPE=//p')"
  [ -n "$KV_EFFECTIVE_DTYPE" ] || die "empty effective KV dtype -- cannot derive the cache namespace"
  # Review fix (round 24): SINGLE SOURCE OF TRUTH for the engine args -- the
  # engine is started with the SAME values the namespace was hashed from.
  # A conflicting KVP_* export (e.g. KVP_KV_CACHE_DTYPE=fp8 with
  # KV_CACHE_DTYPE=bfloat16) used to let namespace+gate agree on one value
  # while docker ran vLLM with the raw operator name -- a desync the resolver
  # alone could not prevent. Sync the operator vars from the resolved values
  # so every consumer (docker args, smoke request, namespace, gate) agrees.
  SERVED_MODEL_NAME="$(printf '%s\n' "$_EFF_ENV" | sed -n 's/^KVP_SERVED_MODEL_NAME=//p')"
  KV_CACHE_DTYPE="$KV_EFFECTIVE_DTYPE"
  MAX_MODEL_LEN="$(printf '%s\n' "$_EFF_ENV" | sed -n 's/^KVP_MAX_MODEL_LEN=//p')"
  KV_PERSIST_BLOCK_SIZE="$(printf '%s\n' "$_EFF_ENV" | sed -n 's/^KVP_BLOCK_SIZE=//p')"
  TP_SIZE="$(printf '%s\n' "$_EFF_ENV" | sed -n 's/^KVP_TP_SIZE=//p')"
  [ -n "$SERVED_MODEL_NAME" ] && [ -n "$MAX_MODEL_LEN" ] && [ -n "$KV_PERSIST_BLOCK_SIZE" ] && [ -n "$TP_SIZE" ] \
    || die "effective identity is incomplete -- refusing to start with desynced engine args"
  echo "KV_PERSIST=1: engine args synced with the effective identity (served-model-name=$SERVED_MODEL_NAME kv=$KV_CACHE_DTYPE max-len=$MAX_MODEL_LEN block=$KV_PERSIST_BLOCK_SIZE tp=$TP_SIZE)"

  # --- cache-identity namespace (identical purpose to kv-persistence round 13)
  KV_PERSIST_NAMESPACE="${KV_PERSIST_NAMESPACE:-}"
  if [ -z "$KV_PERSIST_NAMESPACE" ]; then
    if [ -n "$MODEL_DIR" ] && [ -f "$KVP_REPO/implementation/kvpersist/cli.py" ]; then
      # Review fix: pass the REAL engine settings as KVP_* overrides so the
      # namespace changes with them (fp8 KV must NOT hash as bf16). KVP_* env
      # wins over the shipped identity files in identity_from_env_files().
      # The CLI's stderr is kept: on error we MUST see why (a silent
      # fallback hid "cannot derive LMCache model_name: ..." behind a
      # misleading "kv-persistence not found" message).
      KV_PERSIST_NAMESPACE="$(
        KVP_REPO_ROOT="$KVP_REPO" \
        KVP_SERVED_MODEL_NAME="${KVP_SERVED_MODEL_NAME:-${SERVED_MODEL_NAME:-mimo-pro}}" \
        KVP_KV_CACHE_DTYPE="$KV_EFFECTIVE_DTYPE" \
        KVP_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        KVP_BLOCK_SIZE="$KV_PERSIST_BLOCK_SIZE" \
        KVP_TP_SIZE="${TP_SIZE:-4}" \
        PYTHONPATH="$KVP_REPO/implementation${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m kvpersist.cli model-name 2>"$HERE/kvp-model-name.err" | tail -1 || true
      )"
      if [ -z "$KV_PERSIST_NAMESPACE" ]; then
        echo "KV_PERSIST=1: kv-persistence CLI model-name FAILED at $KVP_REPO:" >&2
        cat "$HERE/kvp-model-name.err" >&2 2>/dev/null || true
        die "cannot derive the cache-identity namespace from kv-persistence.
Fix the error above (usually incomplete identity.env -- run
  bash $KVP_REPO/scripts/setup-identity.sh \"$MODEL_DIR\"
first), or set KV_PERSIST_NAMESPACE explicitly. Refusing to guess: a wrong
namespace could put new KV under an old identity."
      fi
    fi
    if [ -z "$KV_PERSIST_NAMESPACE" ] && [ -n "$MODEL_DIR" ]; then
      # Deterministic fallback: hash everything that changes KV compatibility.
      KV_PERSIST_NAMESPACE="kvp-$(python3 - "$MODEL_DIR" "$KV_EFFECTIVE_DTYPE" "$MAX_MODEL_LEN" "$KV_PERSIST_BLOCK_SIZE" "${TP_SIZE:-4}" <<'PYEOF'
import hashlib, json, os, sys
model_dir, kv_dtype, max_len, block_size, tp_size = sys.argv[1:6]
h = hashlib.sha256()
for name in ("config.json",) + tuple(sorted(
        f for f in os.listdir(model_dir) if f.endswith(".index.json"))):
    p = os.path.join(model_dir, name)
    if os.path.isfile(p):
        h.update(open(p, "rb").read())
for f in sorted(f for f in os.listdir(model_dir)
                if f.endswith((".safetensors", ".bin", ".pt", ".gguf"))):
    p = os.path.join(model_dir, f)
    h.update(f.encode())
    h.update(str(os.path.getsize(p)).encode())
# Review fix: tokenizer files and the chat template also change KV values
# (token ids -> chunk hashes), and TP size changes the per-rank KV layout.
for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
             "special_tokens_map.json", "chat_template.jinja", "chat_template"):
    p = os.path.join(model_dir, name)
    if os.path.isfile(p):
        h.update(name.encode())
        h.update(open(p, "rb").read())
h.update(("|kv=%s|maxlen=%s|blk=%s|tp=%s" % (kv_dtype, max_len, block_size, tp_size)).encode())
print(h.hexdigest()[:16])
PYEOF
)"
      echo "KV_PERSIST=1: kv-persistence not found at $KVP_REPO -- using the fallback namespace hash." >&2
      echo "              The fallback covers weights+tokenizer+template+engine args but NOT the full identity gate;" >&2
      echo "              install kv-persistence (git clone https://github.com/sikkatu/kv-persistence) for the verified one." >&2
    fi
    [ -n "$KV_PERSIST_NAMESPACE" ] || die "KV_PERSIST=1 requires MODEL_DIR (local checkpoint) to derive the cache-identity namespace.
Remote fetch (MODEL_DIR unset) cannot be namespace-isolated: the engine reports the
HF id as model_config.model, so two revisions of one HF id would share one L2
namespace. Use MODEL_DIR=<local checkpoint dir> (same requirement as
kv-persistence's own launcher), or set KV_PERSIST_NAMESPACE explicitly ONLY if you
accept that remote-fetch deployments of the same HF id share cache blocks."
  fi
  # Mount the model under the namespace path so model_config.model is unique
  # per identity (LMCache keys L2 objects by exactly that string).
  if [ -n "$MODEL_DIR" ]; then
    MODEL_MOUNT="/models/$KV_PERSIST_NAMESPACE"
    MOUNTS+=(-v "$MODEL_DIR:$MODEL_MOUNT:ro")
    MODEL_ARGS=(--model "$MODEL_MOUNT")
  fi
  KV_TRANSFER_CONFIG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://$KV_PERSIST_HOST\",\"lmcache.mp.port\":\"$KV_PERSIST_PORT\"}}"
  echo "KV_PERSIST=1: attaching LMCache connector at $KV_PERSIST_HOST:$KV_PERSIST_PORT (wiki: kv-persistence)"
  echo "KV_PERSIST=1: cache-identity namespace -> ${MODEL_MOUNT:-$MODEL_ARG} (incompatible deployments get a safe cache miss)"
  MODEL_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
  # cached_tokens (usage.prompt_tokens_details) is the ONLY machine-readable
  # reuse counter -- without this flag the kv-persistence acceptance run
  # (test-recovery.sh) can never PASS (review fix).
  MODEL_ARGS+=(--enable-prompt-tokens-details)
  # Pin the KV block size explicitly: the kv-persistence identity pin
  # (KVP_BLOCK_SIZE) must match a verifiable engine argument.
  MODEL_ARGS+=(--block-size "$KV_PERSIST_BLOCK_SIZE")
  echo "KV_PERSIST=1: identity-check your deployment BEFORE trusting cached data:"
  echo "  set -a; . $LIVE_DIR/kvp-effective-identity.env; set +a   # SAME effective engine values as this run"
  echo "  export KVP_CACHE_ROOT=...   # the kv-persistence cache root (configs/generic/cache.env)"
  echo "  bash \$KVP_REPO/scripts/verify-engine-identity.sh $CONTAINER   # exit 4/5 = do NOT reuse cache"
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
  --served-model-name "${SERVED_MODEL_NAME:-mimo-pro}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE:-4}" \
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
    # Review fix (round 24): the smoke request must use the ACTUAL served model
    # name (a hard-coded "mimo-pro" got 404 on any other SERVED_MODEL_NAME and
    # failed the launch). The JSON is built by python3 -- the value is safely
    # encoded, never interpolated into JSON by hand.
    # Review fix (round 25): pass the served-model-name EXPLICITLY -- the
    # launcher assigns SERVED_MODEL_NAME from _EFF_ENV without export, so an
    # os.environ lookup in python would fall back to mimo-pro whenever only
    # KVP_SERVED_MODEL_NAME was set (docker args used my-deployment, the smoke
    # request did not). Explicit per-command env makes the smoke body agree
    # with the engine arg in EVERY case.
    SMOKE_BODY="$(SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-pro}" python3 -c 'import json,os,sys; print(json.dumps({"model": os.environ.get("SERVED_MODEL_NAME","mimo-pro"), "messages":[{"role":"user","content":"Reply exactly TP4_OK"}], "max_tokens":64, "temperature":0}))')"
    SMOKE_JSON="$(curl -fsS --max-time 120 -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
      "http://$URLHOST:$PORT/v1/chat/completions" \
      -d "$SMOKE_BODY" || true)"
    echo "$SMOKE_JSON"
    # the answer must actually contain TP4_OK: HTTP 200 + a coherent-looking body is not
    # proof of correct weights (the stock loader produced garbage that looked plausible)
    # a failed check is a FAILED LAUNCH: exit 1 so automation cannot mistake it for success
    if printf '%s' "$SMOKE_JSON" | grep -q '"content":"[^"]*TP4_OK'; then
      echo "smoke test: answer contains TP4_OK -- model output verified"
      echo "Baseline ready (no hot/cold split in this launcher). Verify the API with: bash $HERE/check-tp4-api.sh"
      exit 0
    fi
    echo "ERROR: smoke answer does NOT contain TP4_OK -- weights may be loaded wrong" >&2
    echo "       (compare with a run using PATCH_MODEL_FILES=1; see README Known limits)" >&2
    [ -z "$SMOKE_JSON" ] && echo "       (the generation request itself failed or timed out after 120 s)" >&2
    exit 1
  fi
  sleep "$WAIT_SLEEP"
done
echo "Startup timed out; last logs:" >&2
docker logs --tail 250 "$CONTAINER" >&2 || true
exit 1
