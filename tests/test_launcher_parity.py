"""Round 66: launcher PARITY test -- the KV engine launcher must run the same
docker command as the recipe launcher (review: deferred item 2, step 1).

The round-64/65 bugs (two tp4.env parsers, missing /live mount, key in ps,
.curl-header file) all came from the DUPLICATE between run-vllm-with-persist.sh
and the recipe's run-tp4-hotsplit.sh. This test runs BOTH against a stub
docker/nvidia-smi/curl and compares the final `docker run` lines: the
docker flags, env-file usage and every mount must match.

Skips (not fails) when the sibling repo checkout is unavailable.
"""
from __future__ import annotations
import os
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
SKIP = "kv-persistence checkout not found (parity check needs both launchers)"

STUB = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  docker)
    case "$1" in
      info) exit 0 ;;
      container) exit 1 ;;
      inspect) echo "running exit=0 oom=false"; exit 0 ;;
      logs) echo stub-logs; exit 0 ;;
      run)
        case "$*" in
          *--rm*)
            echo "/usr/local/lib/python3.12/dist-packages"
            exit 0 ;;
        esac
        echo "STUB_DOCKER_RUN" >> "$STUB_LOG"
        exit 0 ;;
    esac
    exit 0 ;;
  nvidia-smi)
    [ "$1" = "-L" ] && for i in 1 2 3 4; do echo "GPU $i: STUB"; done
    exit 0 ;;
  curl) exit 22 ;;
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
    home = tmp_path / "home"
    home.mkdir()
    live = tmp_path / "live"
    live.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["MODEL_DIR"] = str(model)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(live)
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(home / ".config" / "kvp" / "api_key")
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    return env, log


def _docker_run_line(log: Path):
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            return line
    return None


def _norm(line: str, tmp_tokens) -> str:
    """Neutralize path tokens that legitimately differ between the two
    launchers (tmp dirs, cache root) before comparing."""
    for tok in tmp_tokens:
        line = line.replace(tok, "<TMP>")
    return line


def test_kv_launcher_matches_the_recipe_docker_run(stub_env, tmp_path):
    """The engine launch must carry the SAME docker flags and mounts on both
    paths: --cap-add IPC_LOCK/SYS_NICE, --compilation-config FULL_DECODE_ONLY,
    the /w hotsplit machinery, the /live counters mount and the key via
    --env-file (never -e VLLM_API_KEY=<value>)."""
    if KV_REPO is None or not (KV_REPO / "scripts" / "run-vllm-with-persist.sh").is_file():
        pytest.skip(SKIP)
    env, log = stub_env

    def run(script, extra):
        if log.exists():
            log.unlink()
        e = {**env, **extra}
        subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=120)
        return _docker_run_line(log)

    common = {
        "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
        "KVP_SAME_UID": "0", "KVP_SKIP_GPU_CHECK": "1", "KVP_HOTSPLIT": "1",
        # the KV launcher resolves the recipe via RECIPE_DIR (start-all.sh and
        # kvp up export it) -- pass it the same way, since HOME is isolated here
        "RECIPE_DIR": str(RECIPE),
        # the parity check compares the DOCKER COMMAND (flags, env, mounts),
        # not the cache identity -- skip the running-cache pre-flight
        "KVP_PERSIST_ENABLED": "false",
    }
    recipe_line = run(RECIPE / "scripts" / "run-tp4-hotsplit.sh", common)
    kv_line = run(KV_REPO / "scripts" / "run-vllm-with-persist.sh", common)
    assert recipe_line, "recipe launcher did not reach docker run"
    assert kv_line, "KV launcher did not reach docker run"

    tokens = (str(tmp_path),)
    r = _norm(recipe_line, tokens)
    k = _norm(kv_line, tokens)

    # The must-haves on BOTH sides (parity of the hardware-verified profile):
    for flag in ("--cap-add IPC_LOCK", "--cap-add SYS_NICE",
                 "--compilation-config", "/w/hotsplit.py", "/w/sitecustomize.py"):
        assert flag in r, f"recipe docker run is missing {flag}"
        assert flag in k, f"KV docker run is missing {flag} (parity broken)"
    # The review's explicit trio (round 67): the /live counters mount, the
    # --env-file key transport and NO key in argv.
    assert ":/live" in r and ":/live" in k, "the /live counters mount must exist on BOTH sides"
    assert "--env-file" in r and "--env-file" in k, "the key must travel via --env-file on BOTH sides"
    for side, name in ((r, "recipe"), (k, "KV")):
        assert "VLLM_API_KEY=" not in side, f"{name}: the key must never be an argv value"
    # The key must NEVER be a command-line argument on either side
    assert "VLLM_API_KEY=" not in r.split("--env-file")[0], "recipe: key on the command line"
    assert "VLLM_API_KEY=" not in k.split("--env-file")[0], "KV: key on the command line"
    # The hotsplit env must be identical on both sides
    for var in ("HOTSPLIT_DYNAMIC=", "HOTSPLIT_NUMA=", "HOTSPLIT_BOOT_HOOK=",
                "HOTSPLIT_COUNTS=", "HOTSPLIT_HOST_FLOOR_GIB="):
        assert var in k, f"KV docker run is missing {var} (the shared module must provide it)"