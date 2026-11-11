#!/usr/bin/env bash
# MiMo-V2.6-Pro on 4x RTX PRO 6000: vLLM TP4 + UVA expert offload + hotsplit (static or dynamic hot experts).
# VERIFIED (2026-10-23) on target hardware with MOE_BACKEND=humming: static AND dynamic (hit 41.3->45.4%, 11.0 tok/s at PCIe x16 2026-10-25; 5-query A/B vs baseline showed only cosmetic differences -- full regression evaluation pending). Marlin + UVA still OOMs at repack. Read ../README.md first.
set -Eeuo pipefail
die(){ echo "ERROR: $*" >&2; exit 1; }
. "$(dirname "${BASH_SOURCE[0]}")/kvp-common.sh"
# Round 65: send the API key on a stdin pipe (printf is a builtin -- the key is
# never visible in `ps` and no key file is needed). Without a key the request
# goes out unauthenticated (the documented ALLOW_NO_AUTH loopback mode).
# --- seed settings from ./tp4.env (optional) ------------------------------------------------------
# Same contract as manage.sh: values already in the environment WIN over the file -- the file
# is applied only for names that are not already exported ("only fill values not already
# exported"). Copy tp4.env.example to tp4.env and edit; no 'set -a' sourcing (that would lose
# the env-priority contract). The file is read before any validation so IMAGE / MODEL_REVISION /
# API_KEY set in it are honoured.
# tp4.env seeding is now _kvp_seed_env_file from kvp-common.sh (round 67).
_kvp_seed_env_file "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tp4.env"

# --- security contract (SECURITY.md): no wide bind, no disabled auth -----------------------------
# BIND is where the SERVER listens (host networking: -p would be a no-op, so the bind is the only
# control). Loopback by default; API_KEY is required unless you explicitly opt out.
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
# Round 56 (user): lossless compression ON by default -- no exports needed.
# The kv-persistence cache (LMCACHE_L2_SERDE) folds this; opt out:
#   LMCACHE_L2_SERDE=none  (byte-exact raw) / =lz4 (faster, larger)
# Round 64 (review): export the default codec ONLY when none of the documented
# opt-out switches is set. With the unconditional export (round 63), a
# USE_LOSSLESS_SERDE=0 or KVP_L2_SERDE=none in tp4.env was overridden by the
# exported zstd default. Precedence: explicit env switch > pin > default.
if [ -z "${LMCACHE_L2_SERDE:-}" ] && [ -z "${KVP_L2_SERDE:-}" ] && [ -z "${USE_LOSSLESS_SERDE:-}" ]; then
  LMCACHE_L2_SERDE=zstd
fi
# Export the effective codec so the container env and the kv-persistence
# identity gate read the SAME value (both read LMCACHE_L2_SERDE). When the
# user pinned a different codec (KVP_L2_SERDE / USE_LOSSLESS_SERDE) the fold
# in kv-persistence resolves it; here we only pass the raw configured value.
export LMCACHE_L2_SERDE="${LMCACHE_L2_SERDE:-}"
LIVE_DIR="${LIVE_DIR:-$PWD/live}"
# Round 56 (user): ONE API key for BOTH repos (kv-persistence + recipes).
# Canonical file: $HOME/.config/kvp/api_key. A legacy per-repo live/api_key
# is migrated here so old installs keep working -- but new keys are written
# to the canonical path (and mirrored into $LIVE_DIR for older tooling).
KEY_FILE="${KVP_API_KEY_FILE:-$HOME/.config/kvp/api_key}"
mkdir -p "$LIVE_DIR" "$(dirname "$KEY_FILE")"
if [ ! -f "$KEY_FILE" ] && [ -f "$LIVE_DIR/api_key" ]; then
  cp "$LIVE_DIR/api_key" "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "API key migrated to the shared location: $KEY_FILE"
fi
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
    printf '%s\n' "$API_KEY" > "$LIVE_DIR/api_key"
    chmod 600 "$LIVE_DIR/api_key"
    echo "API key auto-generated for the loopback server and saved to $KEY_FILE"
    echo "(not printed here; check-tp4-api.sh and manage.sh read it automatically)."
  fi
fi
# Round 78 (review 2): the canonical key file is the source of truth for
# persistent tooling. An explicit API_KEY from the environment WINS for this
# launch AND is saved to the canonical file (so the two can never diverge).
# Without this, the old code left ~/.config/kvp/api_key pointing at the OLD
# key while live/api_key mirrored the NEW one -- a split-brain.
if [ -n "$API_KEY" ]; then
  _canon=""
  [ -f "$KEY_FILE" ] && _canon="$(head -n1 "$KEY_FILE" | tr -d '[:space:]')"
  if [ "$_canon" != "$API_KEY" ]; then
    if [ -n "$_canon" ]; then
      echo "WARNING: API_KEY from the environment REPLACES the canonical key in $KEY_FILE." >&2
      echo "         All tooling reading that file will use the new key from now on." >&2
    fi
    umask 077
    printf '%s\n' "$API_KEY" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
  fi
fi
# Round 58: ALWAYS keep live/api_key in sync with the canonical key. Older
# tooling (manage.sh, check-tp4-api.sh, quality-compare.sh, bench scripts)
# reads live/api_key; when the shared key already existed these tools saw no
# key and reported "API requires a key" although the key exists. Mirror it
# on EVERY launch (create or refresh). Security is unchanged: the key file is
# chmod 600 and gitignored; nothing new is exposed.
if [ -n "$API_KEY" ]; then
  umask 077
  _cur=""
  [ -f "$LIVE_DIR/api_key" ] && _cur="$(head -n1 "$LIVE_DIR/api_key" | tr -d '[:space:]')"
  if [ "$_cur" != "$API_KEY" ]; then
    printf '%s\n' "$API_KEY" > "$LIVE_DIR/api_key"
    chmod 600 "$LIVE_DIR/api_key"
  fi
fi

# --- contract: image digests, always (no :latest, no floating tags) -------------------------------
# The model patches below are built against a specific vLLM image, so the image MUST be pinned.
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
  *) die "IMAGE must be pinned by digest (@sha256:...), got: '${IMAGE:-<empty>}'. Floating tags are not
reproducible and the mounted model patches only match one vLLM build. Get the digest first:
  docker buildx imagetools inspect <repo>:<tag> | grep Digest
then run: IMAGE=<repo>:<tag>@sha256:<digest> $0" ;;
esac
# Round 80 review (diagnostics parity): a DIRECT run-tp4-hotsplit.sh (the
# README documents it as the alternative to start-all.sh) must run the SAME
# pre-flights the integrated path runs -- the TCP reachability, L2 codec and
# lmcache-version checks (env.sh::_kvp_pre_flights) before the ~15-min model
# load. Guarded by KVP_PREFLIGHTS_DONE so start-all.sh (which already ran
# them) does not pay the probe twice.
# Round 81 (review): resolve KVP_REPO EARLY, using the same algorithm as the
# identity gate below -- a DIRECT run without an exported KVP_REPO found
# ~/kv-persistence too late and skipped the pre-flights entirely.
KVP_REPO="${KVP_REPO:-${KVP_REPO_ROOT:-$HOME/kv-persistence}}"   # both spellings accepted
if [ "${KV_PERSIST:-0}" = "1" ] && [ "${KVP_PREFLIGHTS_DONE:-0}" != "1" ]; then
  if [ -f "$KVP_REPO/scripts/env.sh" ]; then
    . "$KVP_REPO/scripts/env.sh"
    echo "KV_PERSIST=1: running the shared pre-flights (TCP, codec, lmcache version)"
    _kvp_pre_flights "$KVP_REPO" "${KV_PERSIST_HOST:-${LMCACHE_HOST:-127.0.0.1}}" "${KV_PERSIST_PORT:-${LMCACHE_PORT:-5555}}" \
      "${KVP_CACHE_ROOT:-/var/lib/kv-persistence/l2}" "$IMAGE" \
      "$(id -u)" "$(id -g)" "${KVP_SAME_UID:-1}" "${KVP_LAUNCH_MODE:-host}" \
      || die "pre-flight failed -- refusing the launch (nothing was started)"
  else
    echo "NOTE: kv-persistence checkout not found (set KVP_REPO) -- shared pre-flights skipped" >&2
  fi
fi
# Round 80 review: remember the VALIDATED image in live/image -- the
# zero-config auto-resolve (print-recipe-image.sh) lived ONLY in the deleted
# kv launcher. Written AFTER the digest check so an invalid IMAGE never
# lands in live/image (test-recovery.sh scenario B reads it). LIVE_DIR comes
# from start-all.sh (exported) or the recipe's own default.
if [ -n "$IMAGE" ]; then
  printf '%s
' "$IMAGE" > "$LIVE_DIR/image" 2>/dev/null || true
fi

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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$(cd "$HERE/.." && pwd)"
MODEL_ID="${MODEL_ID:-XiaomiMiMo/MiMo-V2.6-Pro-MOPD}"
MODEL_DIR="${MODEL_DIR:-}"
CONTAINER="${CONTAINER:-mimo-pro-tp4-hotsplit}"
SITE="${SITE:-}"   # empty = auto-detect inside the image; set explicitly to override
PATCH_MODEL_FILES="${PATCH_MODEL_FILES:-1}"   # 0 = keep the image's own mimo_v2.py; hook via sitecustomize
PORT="${PORT:-30007}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"   # 1M context by default (model max_position_embeddings=1048576)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"        # auto = the model default (bf16 for MiMo; the VERIFIED benchmarks ran on auto); fp8 halves KV memory (use for long context)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"   # long-context prefill (1M context default)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-128}"        # per rank; with hotsplit it only decides the initial placement
GPU_UTIL="${GPU_UTIL:-0.85}"
MOE_BACKEND="${MOE_BACKEND:-humming}"       # humming boots with UVA offload (verified). marlin + UVA OOMs at the repack
#                # (observed 2026-10-22) and is refused unless HOTSPLIT_ALLOW_MARLIN_UVA=1.
#                # NOTE: the full split bank code is Marlin-specific; on humming the split is an experiment.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# hotsplit knobs (documented in tp4.env.example)
HOTSPLIT_COUNTS="${HOTSPLIT_COUNTS:-/w/expert_hist_mix.json}"
HOTSPLIT_WEIGHTS="${HOTSPLIT_WEIGHTS:-decode=1,prefill=0}"
HOTSPLIT_HOT_GIB="${HOTSPLIT_HOT_GIB:-}"      # per rank; empty = same GPU-memory bytes as the stock placement
HOTSPLIT_DRYRUN="${HOTSPLIT_DRYRUN:-0}"
HOTSPLIT_DYNAMIC="${HOTSPLIT_DYNAMIC:-1}"        # 1 = dynamic hot set (default, verified 2026-10-23); 0 = static
# validate/normalize boolean-ish knobs: accept 1/true/yes/on as true, 0/false/no/off as false;
# anything else is a fatal config error (silent misinterpretation previously caused
# HOTSPLIT_DYNAMIC=true to run static and HOTSPLIT_DRYRUN=true to run a real split).
normalize_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) echo 1 ;;
    0|false|FALSE|no|NO|off|OFF|'') echo 0 ;;
    *)
      echo "ERROR: ${2} must be 1/0 (also accepts true/false, yes/no, on/off), got '${1}'" >&2
      exit 1
      ;;
  esac
}
HOTSPLIT_DRYRUN="$(normalize_bool "${HOTSPLIT_DRYRUN:-0}" HOTSPLIT_DRYRUN)"
HOTSPLIT_DYNAMIC="$(normalize_bool "${HOTSPLIT_DYNAMIC:-1}" HOTSPLIT_DYNAMIC)"
HOTSPLIT_NUMA="$(normalize_bool "${HOTSPLIT_NUMA:-1}" HOTSPLIT_NUMA)"
HOTSPLIT_ALLOW_STATIC_FALLBACK="$(normalize_bool "${HOTSPLIT_ALLOW_STATIC_FALLBACK:-0}" HOTSPLIT_ALLOW_STATIC_FALLBACK)"
# HOTSPLIT_HOT_GIB must be a plain integer GiB count: '30G' or '30 GiB' would previously
# surface as a crash from inside the container ~13 minutes into weight loading.
if [ -n "${HOTSPLIT_HOT_GIB:-}" ]; then
  case "$HOTSPLIT_HOT_GIB" in
    [0-9]|[0-9][0-9]|[0-9][0-9][0-9]) : ;;
    *)
      echo "ERROR: HOTSPLIT_HOT_GIB must be a whole number of GiB (e.g. 30), got '$HOTSPLIT_HOT_GIB'" >&2
      exit 1
      ;;
  esac
fi

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

# --- contract: verify every mounted patch file against the pinned manifest before launch ----------
# PATCH_MODEL_FILES replaces vllm/.../mimo_v2.py wholesale, so a drifted patch silently changes the
# model. Refuse to launch unless the files on disk match patches/MANIFEST.sha256 exactly.
( cd "$RECIPE/patches" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1 ) \
  || die "patches/ do not match patches/MANIFEST.sha256. The model-replacement patches must be
byte-identical to the reviewed set; regenerate the manifest only after a deliberate change:
  ( cd patches && sha256sum mimo_v2_hotsplit.py mimo_v2_omni.py hotsplit.py hotsplit_plan.py numa_bind.py sitecustomize.py expert_hist_mix.json > MANIFEST.sha256 )"
echo "patch manifest verified against patches/MANIFEST.sha256"
# TRUST BOUNDARY (read this before the first run):
#   1. The manifest protects against accidental edits and casual tampering -- the manifest lives
#      in the same tree as the patches. A hostile actor with write access to the repo can replace
#      both. Pin the git revision you reviewed (git rev-parse HEAD) and diff against it.
#   2. --trust-remote-code lets the model run its own Python (modeling code). Pinning
#      MODEL_REVISION prevents silent changes on later fetches, but it does NOT make the code
#      itself safe. Review the model's remote code before the first run, and run the container
#      with no extra host mounts or secrets beyond what this launcher already sets up.
#   3. The API key is passed as VLLM_API_KEY in the container env. Anyone who can run
#      'docker inspect' on this host can read it. Keep docker access root-equivalent.

for f in "$RECIPE/patches/hotsplit.py" "$RECIPE/patches/hotsplit_plan.py" "$RECIPE/patches/numa_bind.py" \
         "$RECIPE/patches/sitecustomize.py" "$RECIPE/patches/expert_hist_mix.json" \
         "$RECIPE/patches/mimo_v2_hotsplit.py" "$RECIPE/patches/mimo_v2_omni.py"; do
  [ -f "$f" ] || die "missing file: $f"
done

MODEL_ARG="$MODEL_ID"
# Round 79 (review): SAME-UID mode is the DEFAULT (KVP_SAME_UID=1; opt out
# with KVP_SAME_UID=0). The engine container runs as $(id -u):$(id -g) -- the
# SAME user as the cache server on the host. Without it the engine runs as
# root while the server runs as the invoking user, and the server's
# cuIpcOpenEventHandle fails with CUDA_ERROR_MAP_FAILED ("Failed to
# open/create interprocess memory file / remote IPC semaphore pool" =
# EACCES on root-owned cuda.shm.* in /dev/shm) -- the recurring
# register_kv_caches 300 s timeout. This is the SAME contract the kv launcher
# has had since round 45h/48 (run-vllm-with-persist.sh [deleted round 80b] lines 60-75); the
# delegated recipe path (KVP_ENGINE=recipe) lost it. Cache dirs move to
# world-writable /tmp paths because a non-root user cannot write /root.
KVP_SAME_UID="${KVP_SAME_UID:-1}"
ENGINE_USER_ARGS=()
ENGINE_CACHE_ARGS=(-e HF_HOME=/root/.cache/huggingface)
_uid="$(id -u)"; _gid="$(id -g)"
if [ "$KVP_SAME_UID" = "1" ]; then
  ENGINE_USER_ARGS=(--user "$_uid:$_gid")
  # a uid without a passwd entry breaks getpass.getuser() and ~-based caches
  # (torch inductor, flashinfer, vllm usage stats) -- set HOME/USER/LOGNAME
  # explicitly (same as the kv launcher, round 45j).
  _me="$(id -un 2>/dev/null || true)"; _me="${_me:-uid$_uid}"
  ENGINE_CACHE_ARGS=(-e HF_HOME=/tmp/hf-cache -e XDG_CACHE_HOME=/tmp/xdg-cache -e TRITON_CACHE_DIR=/tmp/triton-cache -e HOME=/tmp/engine-home -e "USER=$_me" -e "LOGNAME=$_me")
  MOUNTS=(-v "$HF_CACHE:/tmp/hf-cache")
else
  MOUNTS=(-v "$HF_CACHE:/root/.cache/huggingface")
fi
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
  # find_spec("vllm") returns the PACKAGE dir (.../dist-packages/vllm). The patch mount targets
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
  if [ -z "$SITE" ]; then
    PYBIN=python3
    docker run --rm --entrypoint python3 "$IMAGE" -c 'import sys; print(sys.version)' >/dev/null 2>&1 || PYBIN=python
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
  # compatibility probe: the image's own files must expose the API our patches override
  # The patch overwrites base_loader.process_weights_after_loading (a NAME imported
  # into base_loader's namespace from model_loader/utils.py) -- so verify THAT seam:
  # the name is bound in base_loader, called bare (not utils.<name>()), and defined in utils.py.
  PROBE_CMD='import importlib.util,re,sys
ok=True
def flat(s):
    return re.sub(r"\s+","",s)
# 1) the hook seam: base_loader must bind and bare-call process_weights_after_loading
try:
    bl=importlib.util.find_spec("vllm.model_executor.model_loader.base_loader")
    if bl is None:
        print("MISSING base_loader"); ok=False
    else:
        src=open(bl.origin).read()
        f=flat(src)
        bound=("process_weights_after_loading" in src)
        # bare call: name followed by "(" but not preceded by "." (module-qualified)
        called=bool(re.search(r"(?<![.\w])process_weights_after_loading\(", f))
        if bound and called:
            print("SEAM base_loader->utils.process_weights_after_loading: present (bound + bare-called)")
        else:
            print("SEAM base_loader->utils.process_weights_after_loading: ABSENT (bound=%s called=%s)" % (bound, called)); ok=False
except Exception as e:
    print("ERROR base_loader: %r" % e); ok=False
# 2) the definition must exist in model_loader/utils.py
try:
    u=importlib.util.find_spec("vllm.model_executor.model_loader.utils")
    if u is None:
        print("MISSING model_loader.utils"); ok=False
    elif "def process_weights_after_loading" in open(u.origin).read():
        print("DEF model_loader.utils.process_weights_after_loading: present")
    else:
        print("DEF model_loader.utils.process_weights_after_loading: ABSENT"); ok=False
except Exception as e:
    print("ERROR model_loader.utils: %r" % e); ok=False
# 3) the model files must exist and expose load_weights (the patch replaces them wholesale)
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
  docker run --rm --entrypoint $PYBIN "$IMAGE" -c "$HASH_CMD" || true
  MOUNTS+=(
    -v "$RECIPE/patches/mimo_v2_hotsplit.py:$SITE/vllm/model_executor/models/mimo_v2.py:ro"
    -v "$RECIPE/patches/mimo_v2_omni.py:$SITE/vllm/model_executor/models/mimo_v2_omni.py:ro"
  )
  BOOT_HOOK=0   # the replaced mimo_v2.py already calls hotsplit.apply_hotsplit
fi

# --- preflight guards (OBSERVED on target hardware, 2026-10-22) ------------------------------------
# 1. marlin + UVA offload ALWAYS dies with CUDA OOM: prepare_moe_mxfp4_layer_for_marlin ->
#    repack_weight allocates NEW GPU tensors that --cpu-offload-params does not cover, and they
#    accumulate in VRAM until torch.empty fails (92.7 GiB of 94.97 allocated on rank 3).
#    Refuse to start (saves ~10 min of weight loading before the crash) unless explicitly overridden.
# 2. With full UVA offload every expert tensor is host-resident, so the stock GPU-resident expert
#    bytes are 0 and an empty HOTSPLIT_HOT_GIB would plan an EMPTY hot set -- the split becomes a
#    silent no-op with extra kernel launches. Warn and require an explicit budget.
# 3. The bank construction in patches/hotsplit.py is written against the Marlin kernel's quant-config
#    API (scales passed explicitly). Humming reads scales from the layer object instead, so a full
#    split on humming is NOT equivalent -- only the dry-run path is backend-agnostic.
if [ "$MOE_BACKEND" = "marlin" ] && [ "${HOTSPLIT_ALLOW_MARLIN_UVA:-0}" != "1" ]; then
  die "MOE_BACKEND=marlin + UVA offload is a known CUDA OOM on this hardware (the Marlin repack
creates GPU tensors that --cpu-offload-params does not cover; observed 2026-10-22 on 4x RTX PRO 6000,
92.7 GiB of 94.97 allocated before torch.empty failed). Options:
  * MOE_BACKEND=humming HOTSPLIT_DRYRUN=1   -- boot + plan only (verified path)
  * MOE_BACKEND=humming (full run)          -- EXPERIMENT: the bank code is Marlin-specific;
    humming reads scales from the layer, not per-bank. Output may be wrong -- verify before use.
  * HOTSPLIT_ALLOW_MARLIN_UVA=1             -- run anyway and accept the OOM (debugging only)."
fi
if [ -z "$HOTSPLIT_HOT_GIB" ] && [ "$HOTSPLIT_DRYRUN" != "1" ]; then
  echo "WARNING: HOTSPLIT_HOT_GIB is empty and CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB offloads the whole" >&2
  echo "  expert bank. The stock GPU-resident expert bytes are then 0, so the default budget is" >&2
  echo "  0 GiB and the plan will put EVERYTHING in the cold bank -- the split is a no-op." >&2
  echo "  Set HOTSPLIT_HOT_GIB to the number of GiB per rank you want hot (e.g. 20-50)." >&2
fi
if [ "$MOE_BACKEND" != "marlin" ] && [ "$HOTSPLIT_DRYRUN" != "1" ]; then
  echo "NOTE: the full hotsplit bank construction is written against Marlin kernels." >&2
  echo "  MOE_BACKEND=$MOE_BACKEND bypasses the Marlin repack (good for VRAM). The humming" >&2
  echo "  per-bank expert-count patch (patches/hotsplit.py::_patch_humming_bank) handles the" >&2
  echo "  partial-bank assert; output is still experimental -- verify quality before use." >&2
fi
# Round 59: ONE shared module builds the HotSplit env + docker flags
# (scripts/hotsplit_env.sh) -- identical to the KV launcher's, no duplicates.
# The launcher values seeded above (tp4.env) flow in automatically.
HOTSPLIT_BOOT_HOOK="$BOOT_HOOK"
. "$HERE/hotsplit_env.sh"
# Round 76 (review): the FULL HotSplit config validation lives in the shared
# hotsplit_env.sh (_hotsplit_validate_config) -- the same checks the kv
# launcher runs, plus the NUMA_MAP TP-coverage check (every GPU rank 0..TP-1
# needs an entry). A bad value must fail on the HOST, not after the load.
_hotsplit_validate_config "${TP_SIZE:-4}" \
  || { echo "ERROR: HotSplit configuration is invalid -- fix the values above and rerun (nothing was started)." >&2; exit 1; }
# Round 63 (review): the API key must NOT appear in `ps` output. Write it to
# a chmod-600 env-file for docker --env-file and to a curl config file for the
# bearer header; never pass it on a command line.
umask 077
printf 'VLLM_API_KEY=%s\n' "$API_KEY" > "$LIVE_DIR/.engine-env"
chmod 600 "$LIVE_DIR/.engine-env"
ENVS=(
  -e VLLM_USE_DEEP_GEMM=0
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
  --env-file "$LIVE_DIR/.engine-env"
)
# Round 79: cache dirs come from ENGINE_CACHE_ARGS (same-UID /tmp paths or
# the root-mode /root default) so ONE block owns them -- no drift.
ENVS+=("${ENGINE_CACHE_ARGS[@]}")
ENVS+=("${HOTSPLIT_ENVS[@]}")

# --- model arguments: arrays, never re-parsed shell strings --------------------------------------
MODEL_ARGS=(--model "$MODEL_ARG")
if [ -z "$MODEL_DIR" ]; then
  # remote fetch: pin the checkpoint commit SHA (validated above)
  MODEL_ARGS+=(--revision "$MODEL_REVISION")
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
#      (KVP_REPO / ~/kv-persistence): `python3 -m kvpersist.cli model-name`.
#      The REAL engine arguments are passed through as KVP_* overrides so the
#      namespace follows the runtime values, not the shipped pins. KV_CACHE_DTYPE
#      is RESOLVED first (`auto` -> the model's actual dtype from config.json) so
#      an `auto` launch hashes the same as a `bfloat16` launch of that model
#      (review fix: `KV_CACHE_DTYPE=fp8` used to hash as the shipped bf16 pin --
#      a wrong-hit risk, not a safe miss);
#   2. (review round 35) kv-persistence is now a REQUIRED dependency for
#      KV_PERSIST=1: the checkout must carry KVP_COMPAT_VERSION and the
#      identity gate (verify-engine-identity.sh) runs MANDATORY, pre-launch
#      (--check-config-only) AND post-launch against the real container.
#      The old deterministic fallback namespace hash is REMOVED -- since the
#      mandatory gate it was unreachable dead code.
# The result is a path like /models/<namespace>; the engine then reports that
# path as model_config.model and incompatible deployments can never read each
# other's KV blocks (a safe cache miss -- the namespace covers every
# KV-affecting field, so any difference in one of them lands in a different
# namespace).
KV_PERSIST="${KV_PERSIST:-0}"
if [ "$KV_PERSIST" = "1" ]; then
  KV_PERSIST_HOST="${KV_PERSIST_HOST:-127.0.0.1}"
  KV_PERSIST_PORT="${KV_PERSIST_PORT:-${LMCACHE_PORT:-5555}}"   # round 81: SAME resolution as the pre-flight (a bare LMCACHE_PORT must reach both)
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
  KVP_REPO="${KVP_REPO:-${KVP_REPO_ROOT:-$HOME/kv-persistence}}"   # both spellings accepted
  # Review fix (round 34, P2): verify the kv-persistence checkout is a
  # COMPATIBLE version before relying on its CLI or identity gate. The
  # pin lives in $KVP_REPO/KVP_COMPAT_VERSION (currently "1"). Bump it
  # when the identity hash algorithm or CLI interface changes.
  KVP_COMPAT_EXPECTED="${KVP_COMPAT_VERSION:-1}"
  KVP_COMPAT_FOUND=""
  if [ -f "$KVP_REPO/KVP_COMPAT_VERSION" ]; then
    KVP_COMPAT_FOUND="$(head -1 "$KVP_REPO/KVP_COMPAT_VERSION" | tr -d '[:space:]')"
  fi
  if [ "$KVP_COMPAT_FOUND" != "$KVP_COMPAT_EXPECTED" ]; then
    die "kv-persistence compatibility check FAILED: expected KVP_COMPAT_VERSION=$KVP_COMPAT_EXPECTED, found '${KVP_COMPAT_FOUND:-<missing>}' at $KVP_REPO/KVP_COMPAT_VERSION.
Different versions may compute identity differently or lack the CLI interface.
Update the kv-persistence checkout (git pull) or set KVP_COMPAT_VERSION to the expected value."
  fi

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
  # Round 80b (review): an EXPLICIT namespace override requires CONSENT --
  # the round-72 security contract, preserved from the deleted kv launcher.
  # A stray KV_PERSIST_NAMESPACE / legacy KVP_MODEL_NAME silently redirects
  # the cache namespace (and would collide with another deployment's data);
  # it must be deliberate (KVP_ALLOW_NAMESPACE_OVERRIDE=1) and print a loud
  # WARNING, exactly like the old kv launcher did.
  # KV_PERSIST_NAMESPACE is the recipe's OWN explicit knob -- used verbatim.
  # The consent gate is for the legacy KVP_MODEL_NAME only (round 72 contract:
  # an inherited KVP_MODEL_NAME is a silent namespace redirect and must be
  # deliberate). Round 80b: preserved from the deleted kv launcher.
  if [ -n "${KVP_MODEL_NAME:-}" ]; then
    if [ "${KVP_ALLOW_NAMESPACE_OVERRIDE:-0}" != "1" ]; then
      echo "WARNING: KVP_MODEL_NAME is set but KVP_ALLOW_NAMESPACE_OVERRIDE=1 is missing -- IGNORED." >&2
      echo "         an inherited name would silently redirect the cache namespace." >&2
      echo "         consent explicitly: KVP_ALLOW_NAMESPACE_OVERRIDE=1 KVP_MODEL_NAME=<ns> bash $0" >&2
      unset KVP_MODEL_NAME
    else
      echo "WARNING: cache namespace OVERRIDDEN by consent (KVP_ALLOW_NAMESPACE_OVERRIDE=1): $KVP_MODEL_NAME" >&2
      KV_PERSIST_NAMESPACE="${KV_PERSIST_NAMESPACE:-$KVP_MODEL_NAME}"
    fi
  fi
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
        KVP_REPO="$KVP_REPO" \
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
    # Review fix (round 35, P4): the fallback namespace hash is REMOVED.
    # Since round 34 a missing/incompatible kv-persistence checkout aborts the
    # launch earlier (KVP_COMPAT_VERSION check + mandatory identity gate), so
    # the fallback was unreachable dead code. kv-persistence is now a REQUIRED
    # dependency for KV_PERSIST=1.
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
  # Round 80b (review): persist the EFFECTIVE codec + namespace next to the
  # cache -- this file-writing lived ONLY in the deleted kv launcher
  # (round 46) and vanished with it, while verify-engine-identity.sh still
  # READS it. Without it a plain `bash scripts/verify-engine-identity.sh` in
  # a fresh terminal reports l2_serde identity=none != engine=zstd and refuses
  # cache reuse. Regenerated on every launch.
  _eid_root="${KVP_CACHE_ROOT:-/var/lib/kv-persistence/l2}"
  if [ -w "$_eid_root" ]; then
    {
      echo "# Written by run-tp4-hotsplit.sh at engine launch (round 80b)."
      echo "# The EFFECTIVE cache identity of the running engine."
      echo "# Standalone scripts (verify-engine-identity.sh, test-recovery.sh,"
      echo "# benchmark.sh) read this file when neither USE_LOSSLESS_SERDE nor"
      echo "# the KVP_L2_SERDE identity pin is set. Regenerated on every launch."
      echo "KVP_L2_SERDE=${LMCACHE_L2_SERDE:-none}"
      echo "KVP_MODEL_NAME=${KV_PERSIST_NAMESPACE:-$KVP_MODEL_NAME}"
    } > "$_eid_root/engine-identity.env"
    echo "KV_PERSIST=1: engine identity written: $_eid_root/engine-identity.env (KVP_L2_SERDE=${LMCACHE_L2_SERDE:-none})"
  fi
  MODEL_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
  # cached_tokens (usage.prompt_tokens_details) is the ONLY machine-readable
  # reuse counter -- without this flag the kv-persistence acceptance run
  # (test-recovery.sh) can never PASS (review fix).
  MODEL_ARGS+=(--enable-prompt-tokens-details)
  # Pin the KV block size explicitly: the kv-persistence identity pin
  # (KVP_BLOCK_SIZE) must match a verifiable engine argument.
  MODEL_ARGS+=(--block-size "$KV_PERSIST_BLOCK_SIZE")
  # Review fix (round 34, P1): the identity gate is MANDATORY, exactly
  # like run-vllm-with-persist.sh -- cache reuse is BLOCKED on MISMATCH
  # (exit 4) or INCONCLUSIVE (exit 5). No more manual reminder.
  if [ -f "$KVP_REPO/scripts/verify-engine-identity.sh" ]; then
    echo "KV_PERSIST=1: identity gate (mandatory) -- verifying identity config"
    _gate_rc=0
    set -a; . "$LIVE_DIR/kvp-effective-identity.env" 2>/dev/null || true; set +a
    bash "$KVP_REPO/scripts/verify-engine-identity.sh" --check-config-only \
      || _gate_rc=$?
    if [ "$_gate_rc" -ne 0 ]; then
      if [ "$_gate_rc" -eq 5 ]; then
        die "INCONCLUSIVE: engine identity could not be verified -- cache reuse REFUSED by policy.
Fix configs/.../identity.env(.local) and rerun."
      fi
      die "ENGINE IDENTITY MISMATCH (exit $_gate_rc): cache must NOT be reused.
Fix configs/.../identity.env(.local) and rerun."
    fi
    echo "KV_PERSIST=1: identity gate PASSED -- persistent cache is trusted"
  else
    die "KV_PERSIST=1 requires kv-persistence's verify-engine-identity.sh at $KVP_REPO/scripts/.
Clone https://github.com/sikkatu/kv-persistence (set KVP_REPO if non-standard).
Without the identity gate cache reuse is REFUSED by policy."
  fi
fi
# Round 79: the fail-fast mark is taken BEFORE docker run -- any cache-server
# error from the moment the engine starts is "fresh" (a registration failure
# in the first seconds would be missed by a mark taken after the launch).
_kvp_slog="${KVP_CACHE_ROOT:-/var/lib/kv-persistence/l2}/lmcache-server.log"
_kvp_slog_before=0
if [ -f "$_kvp_slog" ]; then _kvp_slog_before=$(wc -c < "$_kvp_slog"); fi
# Round 80b (review): kv-safety checks preserved from the deleted kv launcher.
# (a) GPU memory must be free BEFORE the engine starts (a leaked process would
#     block the new engine after ~15 min of model loading).
if [ "${KVP_SKIP_GPU_CHECK:-0}" != "1" ] && [ -n "${KVP_REPO:-}" ] && [ -f "$KVP_REPO/scripts/env.sh" ]; then
  . "$KVP_REPO/scripts/env.sh"
  _kvp_gpu_preflight 2048 || die "GPU memory is still held -- leaked processes block the new engine (see PIDs above). Fix and rerun."
fi
# (b) with KV_PERSIST=1 the cache server MUST be alive (a dead server wastes
#     the model load and fails at register_kv_caches after 300s).
if [ "${KV_PERSIST:-0}" = "1" ]; then
  _kvp_pidf="${KVP_CACHE_ROOT:-/var/lib/kv-persistence/l2}/lmcache-server.pid"
  if [ ! -f "$_kvp_pidf" ] || ! kill -0 "$(cat "$_kvp_pidf" 2>/dev/null)" 2>/dev/null; then
    die "LMCache cache server not running -- start it first: bash $KVP_REPO/scripts/start-cache.sh"
  fi
fi
docker run -d --name "$CONTAINER" --restart no \
  --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  ${ENGINE_USER_ARGS[@]+"${ENGINE_USER_ARGS[@]}"} \
  "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" \
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
  ${HOTSPLIT_COMPILE_ARG} \
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
# Round 79 (review): fail-fast on cache-server handler errors. The engine
# waits 300s for register_kv_caches, but the server may have ALREADY failed
# with 'Error handling operation' (e.g. CUDA_ERROR_MAP_FAILED) -- waiting out
# the timeout cannot fix it. Scan only the FRESH part of the server log each
# tick (same contract as the kv launcher, run-vllm-with-persist.sh round 45).
for _ in $(seq 1 "$WAIT_TRIES"); do
  # fail-fast (round 79): a fresh server-side handler error is fatal
  if [ -f "$_kvp_slog" ]; then
    _kvp_serr="$(tail -c "+$(( _kvp_slog_before + 1 ))" "$_kvp_slog" 2>/dev/null | grep -a -m1 'Error handling operation register_kv_cache' || true)"
    if [ -n "$_kvp_serr" ]; then
      echo "" >&2
      echo "ERROR: the cache server FAILED to handle a request (fail-fast):" >&2
      tail -c "+$(( _kvp_slog_before + 1 ))" "$_kvp_slog" | grep -a -A20 'Error handling operation register_kv_cache' | head -25 >&2
      echo "" >&2
      echo "This is the root cause -- waiting 300s cannot fix it. See the exception above." >&2
      echo "Fix (CUDA IPC boundary): run the cache server as the SAME user as the engine" >&2
      echo "  (KVP_SAME_UID=1 is now the default; KVP_SAME_UID=0 for root fallback)," >&2
      echo "  or test the boundary first: bash scripts/ipc-test.sh" >&2
      exit 1
    fi
  fi
  # fail fast: if the container died, waiting cannot help -- show why and stop now
  RUN_STATE="$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$CONTAINER" 2>/dev/null || echo gone)"
  case "$RUN_STATE" in
    running*) : ;;
    *)
      echo "ERROR: container $CONTAINER is not running ($RUN_STATE) -- the engine never started." >&2
      echo "Last logs:" >&2
      docker logs --tail 250 "$CONTAINER" >&2 || true
      # Round 80b (review): the cache-server diagnostics lived ONLY in the
      # legacy kv launcher and vanished with it -- restored here so a dead
      # engine still points at the REAL root cause (server log first).
      _kvp_diag="${KVP_CACHE_ROOT:-/var/lib/kv-persistence/l2}"
      echo "" >&2
      echo "---- cache server diagnostics (automatic) ----" >&2
      _kvp_log="$_kvp_diag/lmcache-server.log"
      _kvp_pidf="$_kvp_diag/lmcache-server.pid"
      if [ -f "$_kvp_pidf" ]; then
        _kvp_pid="$(tr -d '[:space:]' < "$_kvp_pidf")"
        if kill -0 "$_kvp_pid" 2>/dev/null; then
          echo "server process: ALIVE (pid $_kvp_pid)" >&2
        else
          echo "server process: DEAD (stale pid file, pid $_kvp_pid)" >&2
        fi
      else
        echo "server process: no pid file at $_kvp_pidf" >&2
      fi
      if [ -f "$_kvp_log" ]; then
        _kvp_err="$(grep -a -n 'Error handling operation' "$_kvp_log" | tail -n 1 || true)"
        if [ -n "$_kvp_err" ]; then
          echo "--- server ERROR (root cause) ---" >&2
          _kvp_errline="${_kvp_err%%:*}"
          sed -n "${_kvp_errline},$(( _kvp_errline + 25 ))p" "$_kvp_log" >&2 || true
        else
          echo "--- no 'Error handling operation' in the server log ---" >&2
        fi
        echo "--- last 30 server log lines (filtered) ---" >&2
        grep -a -v 'skipping eviction' "$_kvp_log" | tail -n 30 >&2 || true
      else
        echo "server log: missing ($_kvp_log)" >&2
      fi
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
  if kvp_curl -fsS --max-time 3 "http://$URLHOST:$PORT/v1/models" > "$MODELS_JSON" 2>/dev/null; then
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
    # Review fix (round 35, P1): the FULL identity gate now runs AFTER the
    # server is ready and BEFORE the first smoke request -- exactly like
    # run-vllm-with-persist.sh post-launch. --check-config-only (run before
    # docker run) does NOT verify the real engine: the container's actual
    # --kv-cache-dtype / --block-size / TP size, the attached LMCacheMPConnector
    # and the RUNNING LMCache server's L2 codec. MISMATCH (4) or INCONCLUSIVE
    # (5) here STOPS the container and fails the launch -- unverified cache is
    # never trusted.
    if [ "$KV_PERSIST" = "1" ]; then
      echo "KV_PERSIST=1: identity gate (post-launch) -- verifying real engine args + server codec"
      _pgate_rc=0
      bash "$KVP_REPO/scripts/verify-engine-identity.sh" "$CONTAINER" \
        || _pgate_rc=$?
      if [ "$_pgate_rc" -ne 0 ]; then
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        if [ "$_pgate_rc" -eq 5 ]; then
          echo "ERROR: INCONCLUSIVE -- engine identity could not be verified (see output above)." >&2
          echo "Cache reuse is REFUSED by policy. Fix configs/.../identity.env(.local) and rerun." >&2
          exit 1
        fi
        echo "ERROR: ENGINE IDENTITY MISMATCH (exit $_pgate_rc) -- cache must NOT be reused." >&2
        echo "Fix configs/.../identity.env(.local) and rerun." >&2
        exit 1
      fi
      echo "KV_PERSIST=1: identity verified -- persistent cache attached and trusted"
    fi
    SMOKE_BODY="$(SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-pro}" python3 -c 'import json,os,sys; print(json.dumps({"model": os.environ.get("SERVED_MODEL_NAME","mimo-pro"), "messages":[{"role":"user","content":"Reply exactly TP4_OK"}], "max_tokens":64, "temperature":0}))')"
    SMOKE_JSON="$(kvp_curl -fsS --max-time 120 -H 'Content-Type: application/json' \
      "http://$URLHOST:$PORT/v1/chat/completions" \
      -d "$SMOKE_BODY" || true)"
    echo "$SMOKE_JSON"
    # the answer must actually contain TP4_OK: HTTP 200 + a coherent-looking body is not
    # proof of correct weights (the stock loader produced garbage that looked plausible).
    # A failed check is a FAILED LAUNCH: exit 1 so automation cannot mistake it for success.
    if printf '%s' "$SMOKE_JSON" | grep -q '"content":"[^"]*TP4_OK'; then
      echo "smoke test: answer contains TP4_OK -- model output verified"
      # Round 69 (step-2 prep, review checklist): the same hotsplit/NUMA
      # activation check the kv-persistence engine launcher runs (round 59) --
      # without it, delegating the launch here would LOSE the check. A missing
      # marker is a FAILED LAUNCH (exit 1). Opt out: KVP_SKIP_HOTSPLIT_CHECK=1.
      if [ "${KVP_SKIP_HOTSPLIT_CHECK:-0}" != "1" ]; then
        echo "hotsplit activation check:"
        _hs_log="$(docker logs "$CONTAINER" 2>&1 || true)"
        _hs_fail=0
        check_marker() {
          if grep -qF -- "$2" <<<"$_hs_log"; then
            echo "  [ok] $1 -- $3"
          else
            echo "  [MISSING] $1 -- $3" >&2
            _hs_fail=1
          fi
        }
        _bh="${HOTSPLIT_BOOT_HOOK:-0}"
        if [ "$_bh" = "1" ]; then
          check_marker "boot hook" "HOOK STATUS boot_hook=ACTIVE" "the /w machinery loaded at interpreter start"
        else
          check_marker "boot hook (intentionally inactive)" "HOOK STATUS boot_hook=INACTIVE" "the model patches call apply_hotsplit themselves"
        fi
        if [ "${HOTSPLIT_DYNAMIC:-1}" = "1" ]; then
          check_marker "dynamic hook" "step hook installed" "dynamic hot/cold mode is active"
          check_marker "live counters" "live counter:" "routing counters are collected (basis for the plan)"
          if grep -q -- "live snapshot failed" <<<"$_hs_log"; then
            echo "  [ERROR] live counter snapshots are failing (is /live mounted and writable?)" >&2
            _hs_fail=1
          fi
        fi
        if [ "${HOTSPLIT_NUMA:-1}" = "1" ]; then
          check_marker "numa hook" "HOOK STATUS numa_bind=ACTIVE" "NUMA binding installed (on a VM HOTSPLIT_NUMA_MAP is required)"
        fi
        if grep -q -- "no eligible MXFP4 MoE layers found" <<<"$_hs_log"; then
          echo "  [ERROR] no MoE layers were split -- the hot bank would be empty" >&2
          _hs_fail=1
        fi
        if grep -qE -- "commit failed|rollback ALSO failed" <<<"$_hs_log"; then
          echo "  [ERROR] hotsplit transaction errors in the engine log" >&2
          _hs_fail=1
        fi
        if [ "$_hs_fail" = "0" ]; then
          echo "hotsplit activation: VERIFIED (hooks active, no transaction errors)."
        else
          echo "hotsplit activation: NOT VERIFIED -- the API answers but the hot-expert machinery is not confirmed." >&2
          echo "Check the engine log: docker logs $CONTAINER | grep -i hotsplit" >&2
          exit 1
        fi
      fi
      echo "Now run: $HERE/check-hotsplit-tp4.sh $CONTAINER"
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