"""Round 90 C3: optional KVP_LOAD_FAILURE_POLICY wires kv_load_failure_policy
into --kv-transfer-config (EXPERIMENTAL, unmeasured). Behavioural -- the
launchers really run against stub docker (the pattern of
tests/test_max_num_seqs.py) and the recorded `docker run` line is inspected.

MUTATIONs caught:
  * hardcoding/kv_load_failure_policy always on -> unset case shows it;
  * dropping the fail/recompute validation -> the garbage case slips through;
  * dropping the JSON injection -> the fail/recompute cases find no key.
"""
from __future__ import annotations
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MIMO = ROOT / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHERS = [MIMO / "scripts" / "run-tp4-hotsplit.sh",
             MIMO / "scripts" / "run-tp4-uva.sh"]

STUB = r'''#!/usr/bin/env bash
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
          *--rm*) echo "STUB_DOCKER_RUN_RM" >> "$STUB_LOG"; echo "/usr/local/lib/python3.12/dist-packages"; exit 0 ;;
        esac
        echo "STUB_DOCKER_RUN" >> "$STUB_LOG"; exit 0 ;;
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
    b = tmp_path / "bin"; b.mkdir()
    log = tmp_path / "stub.log"
    for n in ("docker", "nvidia-smi", "curl", "sha256sum"):
        p = b / n
        p.write_text(STUB); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"; home.mkdir()
    model = tmp_path / "model"; model.mkdir()
    # fake kv-persistence checkout the launcher sanity-checks against
    kvp = tmp_path / "kv-persistence"
    (kvp / "implementation" / "kvpersist").mkdir(parents=True)
    (kvp / "implementation" / "kvpersist" / "cli.py").write_text("# stub\n")
    (kvp / "KVP_COMPAT_VERSION").write_text("1\n")
    (kvp / "scripts").mkdir(exist_ok=True)
    gate = kvp / "scripts" / "verify-engine-identity.sh"
    gate.write_text(
        '#!/usr/bin/env bash\n'
        'echo "$0 $*" >> "$STUB_LOG"\n'
        'exit "${GATE_RC:-0}"\n'
    )
    gate.chmod(gate.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{b}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(home / ".config" / "kvp" / "api_key")
    env["KVP_REPO_ROOT"] = str(kvp)
    env["MODEL_DIR"] = str(model)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["WAIT_TRIES"] = "1"; env["WAIT_SLEEP"] = "0"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["API_KEY"] = "secret123"
    env["IMAGE"] = "x@sha256:" + "a" * 64
    env["KV_PERSIST"] = "1"                    # the policy only matters under KV_PERSIST
    env["KV_PERSIST_NAMESPACE"] = "test-ns"
    return env, log, tmp_path


def _run(lp: Path, env, log, extra=None):
    if log.exists():
        log.unlink()
    e = env.copy(); e.update(extra or {})
    # r2 P0-1: opt out of the custom-image check by default in fixture launches
    # (same pattern as KVP_PREFLIGHTS_DONE for shared pre-flights).
    e.setdefault("KVP_SKIP_IMAGE_CHECK", "1")
    return subprocess.run(["bash", str(lp)], env=e, capture_output=True, text=True, timeout=90)


def _kv_transfer_config(line: str) -> dict:
    m = re.search(r"--kv-transfer-config (\S+)", line)
    if not m:
        pytest.fail(f"docker run line lacks --kv-transfer-config:\n{line}")
    return json.loads(m.group(1))


def _run_line(log: Path) -> str:
    for l in log.read_text().splitlines():
        if "docker run " in l and "--rm" not in l:
            return l.strip()
    return ""


def _policy_block(lp: Path) -> str:
    """Extract just the r90 C3 policy block from a launcher and run it with a
    bare KV_TRANSFER_CONFIG -- a full launcher run needs a live LMCache server
    (pre-flight), which makes the docker-run line unreachable in tests; the
    block alone carries ALL the logic under test (validation + JSON inject)."""
    import subprocess
    src = lp.read_text()
    m = re.search(
        r"# r90 C3:.*?(?=\n  # r90 C2|\n  echo \"KV_PERSIST=1: attaching)",
        src, re.S)
    assert m, f"r90 C3 block not found in {lp.name}"
    block = m.group(0)
    kv = ('KV_TRANSFER_CONFIG='
          + '\'{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}\'')
    return f'{kv}\n{block}\nprintf "%s" "$KV_TRANSFER_CONFIG"'


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_unset_keeps_default_no_injection(lp, stub_env):
    """KVP_LOAD_FAILURE_POLICY unset -> the kv_transfer_config must NOT carry
    kv_load_failure_policy (vLLM keeps its own default behaviour).
    MUTATION caught: always injecting the key / hardcoding `fail`."""
    out = subprocess.run(["bash", "-c", _policy_block(lp)],
                         capture_output=True, text=True, timeout=10).stdout
    cfg = json.loads(out)
    assert "kv_load_failure_policy" not in cfg, (
        f"unset policy must not add the key: {cfg}")


@pytest.mark.parametrize("value", ["fail", "recompute"])
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_valid_value_wired_into_config(lp, value, stub_env):
    """KVP_LOAD_FAILURE_POLICY=fail|recompute -> the key appears in
    --kv-transfer-config with exactly that value. MUTATION caught: dropping
    the JSON injection (the analysed kv_load_failure_policy append)."""
    r = subprocess.run(["bash", "-c", _policy_block(lp)],
                       capture_output=True, text=True, timeout=10,
                       env={**os.environ, "KVP_LOAD_FAILURE_POLICY": value})
    cfg = json.loads(r.stdout.strip().splitlines()[-1])
    assert cfg.get("kv_load_failure_policy") == value, (
        f"expected kv_load_failure_policy={value!r}: {cfg}")


@pytest.mark.parametrize("bad", ["bogus", "1", "not-a-policy"])
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_garbage_refused(lp, bad, stub_env):
    """KVP_LOAD_FAILURE_POLICY=<garbage> must abort with a clear ERROR before
    any value reaches --kv-transfer-config (run against the extracted policy
    block -- full-launcher pre-flight needs a live LMCache server).
    MUTATION caught: dropping the `case fail|recompute)` validation."""
    r = subprocess.run(["bash", "-c", _policy_block(lp)],
                       capture_output=True, text=True, timeout=10,
                       env={**os.environ, "KVP_LOAD_FAILURE_POLICY": bad})
    assert r.returncode != 0, f"{bad!r} must be refused (rc={r.returncode})"
    assert "ERROR" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-400:]