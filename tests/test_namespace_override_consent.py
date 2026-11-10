"""Round 76: namespace-override CONSENT test (review: the round-73 security
fix was not covered by any test).

Contract (round 73): KVP_MODEL_NAME from the environment is the same escape
hatch as the recipe's KV_PERSIST_NAMESPACE, but the namespace IS the
model-separation guard. The override therefore requires EXPLICIT consent
(KVP_ALLOW_NAMESPACE_OVERRIDE=1):
  * without the consent flag an inherited KVP_MODEL_NAME is IGNORED with a
    loud WARNING (the identity-derived name wins);
  * with the flag the override applies and a WARNING is still printed.

The consent check lives where the namespace is used -- the persistent-cache
path -- so these tests run the launcher with the connector ON (same stub
harness as the parity test: stub docker, a "running" cache pid, a listening
socket for the reachability pre-flight, real identity pins written and
restored).
"""
from __future__ import annotations
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _contract as _T
import os
import re
import socket
import stat
import subprocess
from pathlib import Path
import pytest

RECIPES = Path(__file__).resolve().parent.parent
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
KV_REPO = next(
    (c for c in (
        RECIPES / "kv-persistence",
        Path.home() / "kv-persistence",
        RECIPES.parent / "kv-persistence",
    ) if c.is_dir()),
    None,
)
SKIP = "kv-persistence checkout not found (consent test needs the kv launcher)"
LAUNCHER = _T.recipe_launcher()  # Round 80b: the recipe owns the engine launch

GOOD_MARKERS = """HOOK STATUS boot_hook=INACTIVE
hotsplit dynamic: step hook installed on GPUModelRunnerImpl (update every 2000 steps, <=2 swaps/layer, margin 1.25)
hotsplit live counter: 69 layers -> /live/counts.json
HOOK STATUS numa_bind=ACTIVE
"""

STUB = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;
      inspect) echo "running exit=0 oom=false"; exit 0 ;;
      logs) printf '%s' "$STUB_DOCKER_LOGS" ;;
      ps) exit 0 ;;
      run)
        case "$*" in
          *--rm*) echo "/usr/local/lib/python3.12/dist-packages"; exit 0 ;;
        esac
        echo "STUB_DOCKER_RUN" >> "$STUB_LOG"
        exit 0 ;;
    esac
    exit 0 ;;
  nvidia-smi) [ "$1" = "-L" ] && for i in 1 2 3 4; do echo "GPU $i: STUB"; done; exit 0 ;;
  curl)
    case "$*" in
      *"/v1/models"*) echo '{"data":[{"id":"mimo-pro"}]}' ;;
      *"/v1/chat/completions"*) echo '{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}' ;;
      *) echo '{}' ;;
    esac
    exit 0 ;;
  sha256sum) exec /usr/bin/sha256sum "$@" ;;
  *) exit 0 ;;
esac
'''


@pytest.fixture()
def stub_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "stub.log"
    for name in ("docker", "nvidia-smi", "curl", "sha256sum"):
        p = bin_dir / name
        p.write_text(STUB)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    home = tmp_path / "home"
    (home / ".config" / "kvp").mkdir(parents=True)
    key_file = home / ".config" / "kvp" / "api_key"
    key_file.write_text("sharedkey123\n")
    key_file.chmod(0o600)
    live = tmp_path / "live"
    live.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["MODEL_DIR"] = str(model)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(live)
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(key_file)
    env["KVP_CACHE_ROOT"] = str(live)
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["KVP_SKIP_IPC_TEST"] = "1"
    env["KVP_SKIP_GPU_CHECK"] = "1"
    env["RECIPE_DIR"] = str(RECIPE)
    env["KVP_REPO"] = str(KV_REPO)
    # a "running" cache server + a listening socket for the pre-flight
    (live / "lmcache-server.pid").write_text(str(os.getpid()))
    env["KVP_SKIP_GPU_CHECK"] = "1"
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(5)
    env["LMCACHE_HOST"] = "127.0.0.1"
    env["LMCACHE_PORT"] = str(port)
    env["KV_PERSIST_PORT"] = env["LMCACHE_PORT"]  # the pre-flight probes the connector port
    # real identity pins (written + restored) so the identity-derived name can
    # be computed when the override is IGNORED
    cfg = KV_REPO / "configs" / "mimo-v2.6-pro"
    local = cfg / "identity.env.local"
    backup = local.read_text() if local.exists() else None
    base = (cfg / "identity.env").read_text()
    for name, value in (("KVP_MODEL_REVISION", "0" * 40),
                        ("KVP_TOKENIZER_REVISION", "0" * 40),
                        ("KVP_CHAT_TEMPLATE_DIGEST", "0" * 64)):
        base = re.sub(r"^(" + name + r")=.*$", r"\1=" + value, base, flags=re.M)
    local.write_text(base)
    yield env, log
    srv.close()
    if backup is None:
        local.unlink(missing_ok=True)
    else:
        local.write_text(backup)


def _launch(env, log, extra):
    if log.exists():
        log.unlink()
    e = {**env, "STUB_DOCKER_LOGS": GOOD_MARKERS, **extra}
    return subprocess.run(["bash", str(LAUNCHER)], env=e,
                          capture_output=True, text=True, timeout=180)


def _model_target(log: Path):
    """The --model value from the recorded docker run (namespace-coded mount)."""
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            m = re.search(r"--model (\S+)", line)
            return m.group(1) if m else None
    return None


def test_override_without_consent_is_ignored(stub_env):
    """KVP_MODEL_NAME WITHOUT KVP_ALLOW_NAMESPACE_OVERRIDE=1 must be IGNORED
    (the identity-derived name wins) and print a loud WARNING."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log = stub_env
    r = _launch(env, log, {"IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
                           "KV_PERSIST": "1",  # Round 80b: the recipe flag (KVP_PERSIST_ENABLED was kv-only)
                           "KVP_MODEL_NAME": "foreign-ns"})
    out = r.stdout + r.stderr
    assert "WARNING" in out and "IGNORED" in out, (
        f"an un-consented override must warn loudly:\n{out[-1200:]}")
    model = _model_target(log)
    assert model and "foreign-ns" not in model, (
        f"the un-consented override must NOT reach the docker run: {model}")
    assert model and "mimo-pro/" in model, (
        f"the identity-derived namespace must win (mimo-pro/<hash>): {model}")


def test_override_with_consent_applies(stub_env):
    """WITH KVP_ALLOW_NAMESPACE_OVERRIDE=1 the override applies (still with a
    WARNING) -- the deliberate escape hatch the parity test uses."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log = stub_env
    r = _launch(env, log, {"IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
                           "KV_PERSIST": "1",  # Round 80b: the recipe flag (KVP_PERSIST_ENABLED was kv-only)
                           "KVP_ALLOW_NAMESPACE_OVERRIDE": "1",
                           "KVP_MODEL_NAME": "parity-ns"})
    out = r.stdout + r.stderr
    assert "WARNING" in out, f"a consented override must still warn:\n{out[-1200:]}"
    model = _model_target(log)
    assert model and "parity-ns" in model, (
        f"the consented override must reach the docker run: {model}")