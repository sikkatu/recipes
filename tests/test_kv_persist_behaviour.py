"""Behavioural tests for the recipe's KV-PERSIST block (review fix).
The earlier contract tests only searched for strings in the launcher text --
they could NOT catch: (1) the namespace ignoring the engine's real arguments
(`KV_CACHE_DTYPE=fp8` hashed as the shipped bf16 pin -> a wrong hit, not a safe
miss), (2) the CLI error being silently swallowed, (3) the two launchers
disagreeing on the namespace prefix. Here the launcher is actually EXECUTED
against stub `docker`/`nvidia-smi`/`curl`/`python3` binaries on PATH: the stub
`python3` records the KVP_* env it received and derives its namespace from it,
so the test asserts the namespace FOLLOWS the real engine arguments, that a
failing CLI aborts the launch instead of guessing, and that the plain
(KV_PERSIST=0) run touches none of it.
"""
from __future__ import annotations
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHERS = [
    BASE / "scripts" / "run-tp4-uva.sh",
    BASE / "scripts" / "run-tp4-hotsplit.sh",
]

# Stub python3: forwards to the real interpreter EXCEPT for the
# `kvpersist.cli model-name` call. There it records the KVP_* env the launcher
# passed and prints a namespace derived from those values (so fp8 vs bfloat16
# MUST yield different namespaces). KV_CLI_FAIL=1 simulates an unusable
# identity config (the CLI's real failure mode with the shipped empty pins).
PYTHON_STUB = '''#!/usr/bin/env bash
case "$*" in
  *kvpersist.cli*model-name*)
    exec /usr/bin/python3 - "$KV_CLI_ENV_LOG" "${KV_CLI_FAIL:-0}" <<'PYEOF'
import hashlib, json, os, sys
log = sys.argv[1] if len(sys.argv) > 1 else ""
fail = len(sys.argv) > 2 and sys.argv[2] == "1"
rec = {k: os.environ.get(k, "") for k in (
    "KVP_KV_CACHE_DTYPE", "KVP_MAX_MODEL_LEN", "KVP_BLOCK_SIZE",
    "KVP_TP_SIZE", "KVP_SERVED_MODEL_NAME")}
if log:
    open(log, "w").write(json.dumps(rec))
if fail:
    sys.stderr.write(
        "cannot derive LMCache model_name: model_revision must be a full "
        "40-hex commit SHA, got ''\\n")
    raise SystemExit(1)
h = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:8]
print("%s/%s" % (rec.get("KVP_SERVED_MODEL_NAME", "mimo-pro"), h))
PYEOF
    ;;
esac
exec /usr/bin/python3 "$@"
'''
DOCKER_STUB = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$1" in
  info) exit 0 ;;
  container) exit 1 ;;
  inspect) echo "${STUB_INSPECT:-running exit=0 oom=false}"; exit 0 ;;
  logs) echo stub-logs; exit 0 ;;
  run)
    case "$*" in
      *--rm*)
        echo "STUB_DOCKER_RUN_RM" >> "$STUB_LOG"
        echo "/usr/local/lib/python3.12/dist-packages"
        exit 0 ;;
    esac
    if [ "${ALLOW_RUN:-0}" = "1" ]; then
      echo "STUB_DOCKER_RUN" >> "$STUB_LOG"
      exit 0
    fi
    echo "STUB_DOCKER_RUN_ATTEMPTED" >> "$STUB_LOG"
    exit 0 ;;
esac
exit 0 ;;
'''
STUB = '''#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$(basename "$0")" in
  nvidia-smi)
    if [ "$1" = "-L" ]; then for i in $(seq 1 "${GPU_COUNT:-4}"); do echo "GPU $i: STUB"; done; fi
    exit 0 ;;
  curl) exit 22 ;;
  sha256sum) exec /usr/bin/sha256sum "$@" ;;
  *) exit 0 ;;
esac
'''


def _install_stubs(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "stub.log"
    cli_env_log = tmp_path / "cli-env.json"
    for name, body in (("docker", DOCKER_STUB), ("python3", PYTHON_STUB),
                       ("nvidia-smi", STUB), ("curl", STUB), ("sha256sum", STUB)):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    # A fake kv-persistence checkout the launcher's namespace CLI runs from.
    kvp = tmp_path / "kv-persistence"
    (kvp / "implementation" / "kvpersist").mkdir(parents=True)
    (kvp / "implementation" / "kvpersist" / "cli.py").write_text("# stub\n")
    # Review (round 34): the launcher pins a COMPATIBLE kv-persistence
    # version (KVP_COMPAT_VERSION) and runs the identity gate MANDATORY.
    (kvp / "KVP_COMPAT_VERSION").write_text("1\n")
    (kvp / "scripts").mkdir(exist_ok=True)
    gate = kvp / "scripts" / "verify-engine-identity.sh"
    # Review (round 35): the gate runs TWICE -- pre-launch (--check-config-only)
    # and post-launch against the real container. GATE_PRE_RC / GATE_RC control
    # the two exit codes independently (GATE_RC alone sets both).
    gate.write_text(
        '#!/usr/bin/env bash\n'
        'echo "$0 $*" >> "$STUB_LOG"\n'
        'case "$*" in\n'
        '  *--check-config-only*) exit "${GATE_PRE_RC:-${GATE_RC:-0}}" ;;\n'
        'esac\n'
        'exit "${GATE_RC:-0}"\n'
    )
    gate.chmod(gate.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    # Review fix (round 25): scrub every name the launcher resolves through the
    # effective-identity code, so each test starts from a known-clean state
    # (an inherited SERVED_MODEL_NAME/KVP_* would silently change what is
    # being tested).
    for _k in ("SERVED_MODEL_NAME", "TP_SIZE", "KV_CACHE_DTYPE", "MAX_MODEL_LEN",
               "KV_PERSIST_BLOCK_SIZE", "BLOCK_SIZE"):
        env.pop(_k, None)
    for _k in [k for k in env if k.startswith("KVP_")]:
        del env[_k]
    env["STUB_LOG"] = str(log)
    env["KV_CLI_ENV_LOG"] = str(cli_env_log)
    env["KVP_REPO_ROOT"] = str(kvp)
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    env["MODEL_DIR"] = str(model_dir)
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["WAIT_TRIES"] = "1"
    # these tests assert identity/override behaviour, not hotsplit
    # activation (their docker stub logs no markers) -- documented opt-out
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    env["WAIT_SLEEP"] = "0"
    return env, log, cli_env_log


def _run(lp: Path, env: dict, extra: dict | None = None):
    e = env.copy()
    e.update(extra or {})
    return subprocess.run(
        ["bash", str(lp)], env=e, capture_output=True, text=True, timeout=120
    )


def _namespace_of(result) -> str | None:
    for line in (result.stdout + result.stderr).splitlines():
        if "cache-identity namespace ->" in line:
            return line.split("->", 1)[1].split("(", 1)[0].strip()
    return None


def _server_runs(log: Path) -> list:
    if not log.exists():
        return []
    return [l for l in log.read_text().splitlines()
            if "docker run " in l and "--rm" not in l and "STUB_DOCKER_RUN\n" in l + "\n"]


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_namespace_follows_real_engine_args(lp, tmp_path):
    """Review fix: `KV_CACHE_DTYPE=fp8` must NOT hash as the shipped bf16 pin.
    The namespace must change with the ACTUAL engine arguments."""
    env, _log, cli_env_log = _install_stubs(tmp_path)
    base = {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
            "KV_PERSIST": "1"}
    r1 = _run(lp, env, dict(base, KV_CACHE_DTYPE="bfloat16"))
    ns1 = _namespace_of(r1)
    assert ns1, f"launcher printed no namespace: {r1.stdout[-1500:]}\n{r1.stderr[-1500:]}"
    rec1 = json.loads(cli_env_log.read_text())
    assert rec1["KVP_KV_CACHE_DTYPE"] == "bfloat16", (
        f"CLI got {rec1['KVP_KV_CACHE_DTYPE']!r}, expected the real engine arg"
    )
    r2 = _run(lp, env, dict(base, KV_CACHE_DTYPE="fp8"))
    ns2 = _namespace_of(r2)
    rec2 = json.loads(cli_env_log.read_text())
    assert rec2["KVP_KV_CACHE_DTYPE"] == "fp8", (
        f"CLI got {rec2['KVP_KV_CACHE_DTYPE']!r} -- the real dtype must reach "
        "the namespace derivation (review: fp8 used to hash as bf16)"
    )
    assert ns2 and ns2 != ns1, (
        "fp8 KV must NOT land in the bfloat16 namespace -- the namespace must "
        f"follow the engine args: {ns1} vs {ns2}"
    )
    for k in ("KVP_MAX_MODEL_LEN", "KVP_BLOCK_SIZE", "KVP_TP_SIZE",
              "KVP_SERVED_MODEL_NAME"):
        assert rec2.get(k), f"{k} missing from the CLI env: {rec2}"


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_cli_failure_aborts_launch_with_real_error(lp, tmp_path):
    """Review fix: `2>/dev/null || true` hid the CLI error behind a misleading
    'kv-persistence not found' fallback. The launcher must FAIL loudly with the
    real reason and must not start the server container."""
    env, log, _cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "KV_PERSIST": "1", "KV_CLI_FAIL": "1"})
    assert r.returncode != 0, "a failing kvpersist.cli model-name must abort the launch"
    combined = r.stdout + r.stderr
    assert "cannot derive LMCache model_name" in combined, (
        f"the real CLI error must be shown, not hidden:\n{combined[-2000:]}"
    )
    assert "kv-persistence not found" not in combined, (
        "the fallback message must NOT be used when the repo IS present"
    )
    assert "STUB_DOCKER_RUN\n" not in (log.read_text() if log.exists() else ""), (
        "the server container must not be started without a namespace"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_namespace_override_is_honoured(lp, tmp_path):
    env, _log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "KV_PERSIST_NAMESPACE": "operator-owned-ns"})
    ns = _namespace_of(r)
    assert ns == "/models/operator-owned-ns", (
        f"explicit KV_PERSIST_NAMESPACE must be used verbatim, got {ns}"
    )
    assert not cli_env_log.exists() or not cli_env_log.read_text(), (
        "the namespace CLI must be skipped when KV_PERSIST_NAMESPACE is set"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_plain_run_has_no_namespace(lp, tmp_path):
    """KV_PERSIST=0 (or unset) must not touch the namespace at all."""
    env, _log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1"})
    assert _namespace_of(r) is None, "plain run must not derive a cache namespace"
    assert not cli_env_log.exists() or not cli_env_log.read_text(), (
        "plain run must not call the namespace CLI"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_kv_persist_block_size_reaches_engine(lp, tmp_path):
    """KV_PERSIST_BLOCK_SIZE (not a generic BLOCK_SIZE) must control --block-size."""
    env, _log, cli_env_log = _install_stubs(tmp_path)
    _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                   "ALLOW_RUN": "1", "KV_PERSIST": "1",
                   "KV_PERSIST_BLOCK_SIZE": "32"})
    rec = json.loads(cli_env_log.read_text())
    assert rec["KVP_BLOCK_SIZE"] == "32", (
        f"KV_PERSIST_BLOCK_SIZE must reach the CLI (got {rec['KVP_BLOCK_SIZE']!r})"
    )
    runs = [l for l in (Path(env["STUB_LOG"]).read_text() if Path(env["STUB_LOG"]).exists() else "").splitlines()
            if "docker run " in l and "--rm" not in l]
    assert any("--block-size 32" in l for l in runs), (
        f"--block-size must pin the requested value, runs: {runs}"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_served_model_name_prefix_is_stable(lp, tmp_path):
    """Review fix: both launchers must agree on the model_name PREFIX so the
    same identity maps to the same namespace (the reviewer measured
    `mimo-pro/<hash>` vs `MiMo-V2.6-Pro-MOPD/<hash>` for the same identity)."""
    env, _log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1"})
    rec = json.loads(cli_env_log.read_text())
    assert rec["KVP_SERVED_MODEL_NAME"] == "mimo-pro", (
        f"KVP_SERVED_MODEL_NAME must default to the served-model-name "
        f"(mimo-pro), not to basename(MODEL_DIR): {rec['KVP_SERVED_MODEL_NAME']!r}"
    )
    ns = _namespace_of(r)
    assert ns and ns.startswith("/models/mimo-pro/"), (
        f"the namespace prefix must be the served-model-name (mimo-pro): {ns}"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_kv_cache_dtype_auto_resolves_to_model_dtype(lp, tmp_path):
    """Review fix: `KV_CACHE_DTYPE=auto` must resolve to the model's EFFECTIVE
    dtype before deriving the namespace, so an `auto` launch of a bf16 model
    lands in the same namespace as an explicit `bfloat16` launch."""
    env, _log, cli_env_log = _install_stubs(tmp_path)
    # The stub model dir has no config.json -> resolution falls back to "auto".
    r1 = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                        "ALLOW_RUN": "1", "KV_PERSIST": "1",
                        "KV_CACHE_DTYPE": "auto"})
    ns_auto = _namespace_of(r1)
    rec_auto = json.loads(cli_env_log.read_text())
    assert rec_auto["KVP_KV_CACHE_DTYPE"], (
        "the CLI must receive a resolved dtype even for 'auto'"
    )
    # With a config.json declaring bfloat16, auto must resolve to bfloat16.
    model_dir = Path(env["MODEL_DIR"])
    (model_dir / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    r2 = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                        "ALLOW_RUN": "1", "KV_PERSIST": "1",
                        "KV_CACHE_DTYPE": "auto"})
    rec2 = json.loads(cli_env_log.read_text())
    assert rec2["KVP_KV_CACHE_DTYPE"] == "bfloat16", (
        f"auto must resolve to the model's dtype (bfloat16), got "
        f"{rec2['KVP_KV_CACHE_DTYPE']!r}"
    )
    # The resolved-auto namespace must equal an explicit-bfloat16 namespace.
    r3 = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                        "ALLOW_RUN": "1", "KV_PERSIST": "1",
                        "KV_CACHE_DTYPE": "bfloat16"})
    assert _namespace_of(r2) == _namespace_of(r3), (
        "an 'auto' launch of a bfloat16 model must land in the SAME namespace "
        "as an explicit 'bfloat16' launch (the namespace must reflect the "
        "EFFECTIVE dtype, not the literal string)"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_namespace_matches_engine_args(lp, tmp_path):
    """Review fix (round 21): SERVED_MODEL_NAME and TP_SIZE used to feed the
    NAMESPACE while the engine was started with the hardcoded constants
    `mimo-pro` / `4`. If someone exported TP_SIZE=2, the namespace claimed
    tp=2 while the engine ran tp=4 -- and could collide with a REAL tp=2
    deployment's cache. Now the engine args follow the same variables: the
    namespace and --tensor-parallel-size / --served-model-name must agree."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "TP_SIZE": "2", "SERVED_MODEL_NAME": "my-deployment"})
    rec = json.loads(cli_env_log.read_text())
    assert rec["KVP_TP_SIZE"] == "2", (
        f"KVP_TP_SIZE must follow TP_SIZE (got {rec['KVP_TP_SIZE']!r})"
    )
    assert rec["KVP_SERVED_MODEL_NAME"] == "my-deployment", (
        f"KVP_SERVED_MODEL_NAME must follow SERVED_MODEL_NAME "
        f"(got {rec['KVP_SERVED_MODEL_NAME']!r})"
    )
    stub = Path(env["STUB_LOG"]).read_text() if Path(env["STUB_LOG"]).exists() else ""
    runs = [l for l in stub.splitlines()
            if "docker run " in l and "--rm" not in l and "STUB_DOCKER_RUN" in stub]
    joined = "\n".join(runs)
    assert "--tensor-parallel-size 2" in joined, (
        f"the ENGINE must run with --tensor-parallel-size 2 (what the "
        f"namespace claims), runs:\n{joined}"
    )
    assert "--served-model-name my-deployment" in joined, (
        f"the ENGINE must run with --served-model-name my-deployment (what "
        f"the namespace claims), runs:\n{joined}"
    )
    ns = _namespace_of(r)
    assert ns and "my-deployment" in ns, (
        f"the namespace must carry the served-model-name: {ns}"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_effective_identity_is_persisted(lp, tmp_path):
    """Review fix (round 23): the launcher must save the EFFECTIVE engine
    identity to live/kvp-effective-identity.env through the shared resolver
    (kvpersist.cli effective-identity), and the MANDATORY identity gate (round
    34) sources that file first -- so the gate compares
    the container args against the SAME values the namespace was hashed from.
    A non-default run (KV_CACHE_DTYPE=fp8 / MAX_MODEL_LEN=524288 / TP=2) must
    not be blocked by MISMATCH against the shipped identity.env pins."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "KV_CACHE_DTYPE": "fp8", "MAX_MODEL_LEN": "524288",
                       "TP_SIZE": "2", "SERVED_MODEL_NAME": "my-dep"})
    eff = Path(env["LIVE_DIR"]) / "kvp-effective-identity.env"
    assert eff.exists(), (
        f"the effective engine identity must be persisted to {eff}:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    vals = dict(l.split("=", 1) for l in eff.read_text().splitlines() if "=" in l)
    assert vals.get("KVP_KV_CACHE_DTYPE") == "fp8", vals
    assert vals.get("KVP_MAX_MODEL_LEN") == "524288", vals
    assert vals.get("KVP_TP_SIZE") == "2", vals
    assert vals.get("KVP_SERVED_MODEL_NAME") == "my-dep", vals
    combined = r.stdout + r.stderr
    assert "kvp-effective-identity.env" in combined, (
        "the launcher must tell the operator about the persisted effective identity"
    )
    # Review round 34: the gate is MANDATORY -- the launcher sources the
    # effective identity file and runs the gate ITSELF (no manual reminder).
    assert "identity gate (mandatory)" in combined, (
        "the launcher must RUN the identity gate, not just mention it:\n"
        f"{combined[-1500:]}"
    )
    assert "identity gate PASSED" in combined, (
        "the mandatory gate must PASS before the cache is trusted:\n"
        f"{combined[-1500:]}"
    )
    assert "verify-engine-identity.sh" in log.read_text(), (
        "the gate script must actually be executed (see the stub log)"
    )




CURL_LOGGING_STUB = """#!/usr/bin/env bash
echo "$0 $*" >> "$STUB_LOG"
case "$*" in
  *"/v1/models"*)
    echo '{"data":[{"id":"probe"}]}'
    exit 0 ;;
  *"/v1/chat/completions"*)
    body=""
    prev=""
    for a in "$@"; do
      [ "$prev" = "-d" ] && body="$a"
      prev="$a"
    done
    echo "curl -d $body" >> "$STUB_LOG"
    echo '{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}'
    exit 0 ;;
esac
exit 22
"""
# ---------------------------------------------------------------------------
# Round 24: conflicting KVP_* vs operator overrides (review item 3).
# ---------------------------------------------------------------------------

def _docker_args(log: Path) -> str:
    if not log.exists():
        return ""
    return "\n".join(l for l in log.read_text().splitlines()
                      if "docker run " in l and "--rm" not in l)


def _smoke_body(log: Path) -> str:
    """The smoke request body captured by the curl stub."""
    if not log.exists():
        return ""
    bodies = []
    for l in log.read_text().splitlines():
        # only the stub's explicit "curl -d <body>" log lines (the generic
        # "$0 $*" line also contains " -d " and would double-capture)
        if l.startswith("curl -d "):
            bodies.append(l[len("curl -d "):].strip())
    return bodies[-1] if bodies else ""


def _run_with_conflict(lp, tmp_path, conflict: dict, docker_check: str):
    """Run the launcher with conflicting env and assert the docker args carry
    the RESOLVED value (docker_check is the expected engine arg substring)."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1", **conflict})
    assert r.returncode == 0, (
        f"launcher must exit 0 with conflicting overrides, got {r.returncode}\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    args = _docker_args(log)
    assert docker_check in args, (
        f"the ENGINE must run with {docker_check!r} (the resolved effective "
        f"value), docker args:\n{args}\nlauncher output:\n"
        f"{r.stdout[-1200:]}\n{r.stderr[-1200:]}"
    )
    return r, log, cli_env_log


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_kv_cache_dtype(lp, tmp_path):
    """KVP_KV_CACHE_DTYPE=fp8 with KV_CACHE_DTYPE=bfloat16: the engine must run
    --kv-cache-dtype fp8 (the KVP_* value the namespace was hashed from)."""
    _run_with_conflict(lp, tmp_path,
                       {"KVP_KV_CACHE_DTYPE": "fp8", "KV_CACHE_DTYPE": "bfloat16"},
                       "--kv-cache-dtype fp8")


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_tp_size(lp, tmp_path):
    """KVP_TP_SIZE=2 with TP_SIZE=4: the engine must run --tensor-parallel-size 2."""
    _run_with_conflict(lp, tmp_path,
                       {"KVP_TP_SIZE": "2", "TP_SIZE": "4"},
                       "--tensor-parallel-size 2")


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_block_size(lp, tmp_path):
    """KVP_BLOCK_SIZE=32 with KV_PERSIST_BLOCK_SIZE=16: the engine must run
    --block-size 32."""
    _run_with_conflict(lp, tmp_path,
                       {"KVP_BLOCK_SIZE": "32", "KV_PERSIST_BLOCK_SIZE": "16"},
                       "--block-size 32")


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_served_model_name(lp, tmp_path):
    """KVP_SERVED_MODEL_NAME=my-deployment with SERVED_MODEL_NAME=mimo-pro: the
    engine must run --served-model-name my-deployment AND the smoke request
    must use my-deployment (not the stale mimo-pro)."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    # a curl stub that logs the request body so the smoke JSON can be checked
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(
        '#!/usr/bin/env bash\n'
        'echo "$0 $*" >> "$STUB_LOG"\n'
        'case "$*" in\n'
        '  *"/v1/models"*)\n'
        '    echo \'{"data":[{"id":"probe"}]}\'\n'
        '    exit 0 ;;\n'
        '  *"/v1/chat/completions"*)\n'
        '    body=""\n'
        '    prev=""\n'
        '    for a in "$@"; do\n'
        '      [ "$prev" = "-d" ] && body="$a"\n'
        '      prev="$a"\n'
        '    done\n'
        '    echo "curl -d $body" >> "$STUB_LOG"\n'
        '    echo \'{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}\'\n'
        '    exit 0 ;;\n'
        'esac\n'
        'exit 22\n'
    )
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "KVP_SERVED_MODEL_NAME": "my-deployment",
                       "SERVED_MODEL_NAME": "mimo-pro"})
    args = _docker_args(log)
    assert "--served-model-name my-deployment" in args, (
        f"the ENGINE must run with the resolved served-model-name, docker args:\n{args}"
    )
    body = _smoke_body(log)
    payload = json.loads(body)
    assert payload["model"] == "my-deployment", (
        "the smoke request must use the resolved served-model-name "
        f"(my-deployment), got model={payload.get('model')!r} in body:\n{body}"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_smoke_without_exported_name(lp, tmp_path):
    """Review fix (round 25): with ONLY KVP_SERVED_MODEL_NAME set (the
    launcher's SERVED_MODEL_NAME is a non-exported local assignment), the
    smoke request must still carry my-deployment -- not fall back to
    mimo-pro via os.environ."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "KVP_SERVED_MODEL_NAME": "my-deployment"})
    assert r.returncode == 0, (
        f"launcher must exit 0, got {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    args = _docker_args(log)
    assert "--served-model-name my-deployment" in args, args
    payload = json.loads(_smoke_body(log))
    assert payload["model"] == "my-deployment", (
        "the smoke request must follow KVP_SERVED_MODEL_NAME even when "
        f"SERVED_MODEL_NAME is not in the environment, got {payload['model']!r}"
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_conflicting_overrides_all_five(lp, tmp_path):
    """All five KVP_* parameters conflict with the operator names at once:
    docker args, the cache-identity namespace and the smoke body must all use
    the KVP_* values (the resolved effective identity)."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    conflict = {
        "KVP_KV_CACHE_DTYPE": "fp8", "KV_CACHE_DTYPE": "bfloat16",
        "KVP_MAX_MODEL_LEN": "524288", "MAX_MODEL_LEN": "1048576",
        "KVP_TP_SIZE": "2", "TP_SIZE": "4",
        "KVP_BLOCK_SIZE": "32", "KV_PERSIST_BLOCK_SIZE": "16",
        "KVP_SERVED_MODEL_NAME": "my-deployment", "SERVED_MODEL_NAME": "mimo-pro",
    }
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1", **conflict})
    assert r.returncode == 0, (
        f"launcher must exit 0, got {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    args = _docker_args(log)
    for check in ("--kv-cache-dtype fp8", "--max-model-len 524288",
                  "--tensor-parallel-size 2", "--block-size 32",
                  "--served-model-name my-deployment"):
        assert check in args, (
            f"docker args must contain {check!r} (the resolved effective value):\n{args}"
        )
    ns = _namespace_of(r)
    assert ns and "my-deployment" in ns, (
        f"the cache-identity namespace must carry the resolved served-model-name: {ns}"
    )
    payload = json.loads(_smoke_body(log))
    assert payload["model"] == "my-deployment", (
        f"the smoke request must carry the resolved served-model-name, got {payload['model']!r}"
    )


# ---------------------------------------------------------------------------
# Round 34 (P1): the identity gate is MANDATORY in the recipe launchers.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_identity_gate_mismatch_blocks_cache(lp, tmp_path):
    """Review round 34, P1: a MISMATCH from the identity gate (exit 4) must
    BLOCK the launch -- same policy as run-vllm-with-persist.sh. Before this
    fix the recipe launchers only PRINTED a reminder to run the gate by hand,
    so vLLM could start with the persistent cache attached and unverified."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1", "GATE_RC": "4"})
    assert r.returncode != 0, (
        "a gate MISMATCH (exit 4) must abort the launch:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert "MISMATCH" in (r.stdout + r.stderr).upper(), (
        "the launcher must report WHY it refused:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert not _server_runs(log), (
        "the engine must NOT be started when the identity gate fails:\n"
        + log.read_text()[-1500:] if log.exists() else ""
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_identity_gate_inconclusive_blocks_cache(lp, tmp_path):
    """Review round 34, P1: INCONCLUSIVE (exit 5) must also block the launch."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1", "GATE_RC": "5"})
    assert r.returncode != 0, (
        "INCONCLUSIVE (exit 5) must abort the launch:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert "INCONCLUSIVE" in (r.stdout + r.stderr).upper(), (
        "the launcher must report the INCONCLUSIVE verdict:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert not _server_runs(log), "the engine must NOT start on INCONCLUSIVE"


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_missing_identity_gate_blocks_cache(lp, tmp_path):
    """Review round 34, P1: without the gate script cache reuse must be
    REFUSED -- no silent fallback to unverified cache."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    gate = Path(env["KVP_REPO_ROOT"]) / "scripts" / "verify-engine-identity.sh"
    gate.unlink()
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1"})
    assert r.returncode != 0, (
        "without the identity gate the launch must be refused:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert "REFUSED" in (r.stdout + r.stderr).upper(), (
        "the launcher must say cache reuse is REFUSED without the gate:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert not _server_runs(log), "the engine must NOT start without the gate"


# ---------------------------------------------------------------------------
# Round 34 (P2): pinned kv-persistence compatibility version.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_incompatible_kvp_version_blocks_launch(lp, tmp_path):
    """Review round 34, P2: a kv-persistence checkout whose
    KVP_COMPAT_VERSION does not match the expected pin must be rejected
    BEFORE anything is launched -- different versions may compute identity
    differently or lack the CLI interface."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    compat = Path(env["KVP_REPO_ROOT"]) / "KVP_COMPAT_VERSION"
    compat.write_text("2\n")
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1"})
    assert r.returncode != 0, (
        "an incompatible kv-persistence version must abort the launch:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    combined = (r.stdout + r.stderr).upper()
    assert "COMPATIBILITY" in combined or "KVP_COMPAT" in combined, (
        "the launcher must name the compatibility check:\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    )
    assert not _server_runs(log), "the engine must NOT start on version mismatch"


# ---------------------------------------------------------------------------
# Round 35 (P1): the FULL identity gate runs POST-LAUNCH, before the smoke
# request. --check-config-only does NOT verify the real engine args (docker
# --kv-cache-dtype / --block-size / TP), the attached connector or the RUNNING
# LMCache server's L2 codec.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lp", LAUNCHERS)
def test_pre_pass_post_mismatch_stops_before_smoke(lp, tmp_path):
    """Review round 35, P1: pre-launch gate PASS + post-launch gate MISMATCH
    (exit 4) -> the container is STOPPED and the smoke request is NEVER sent."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "GATE_PRE_RC": "0", "GATE_RC": "4"})
    combined = r.stdout + r.stderr
    assert r.returncode != 0, (
        f"post-launch MISMATCH must fail the launch:\n{combined[-1500:]}"
    )
    assert "identity gate (post-launch)" in combined, (
        "the post-launch gate must RUN:\n" + combined[-1500:]
    )
    assert "MISMATCH" in combined.upper(), (
        "the MISMATCH verdict must be reported:\n" + combined[-1500:]
    )
    stublog = log.read_text() if log.exists() else ""
    assert "docker rm -f" in stublog, (
        "the container must be STOPPED on post-launch MISMATCH:\n" + stublog[-1500:]
    )
    assert "curl -d " not in stublog, (
        "the smoke request must NOT be sent when the post-launch gate fails:\n"
        + stublog[-1500:]
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_pre_pass_post_inconclusive_stops_before_smoke(lp, tmp_path):
    """Review round 35, P1: pre-launch PASS + post-launch INCONCLUSIVE (exit 5)
    -> the container is stopped and the launch fails."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "GATE_PRE_RC": "0", "GATE_RC": "5"})
    combined = r.stdout + r.stderr
    assert r.returncode != 0, (
        f"post-launch INCONCLUSIVE must fail the launch:\n{combined[-1500:]}"
    )
    assert "INCONCLUSIVE" in combined.upper(), (
        "the INCONCLUSIVE verdict must be reported:\n" + combined[-1500:]
    )
    stublog = log.read_text() if log.exists() else ""
    assert "docker rm -f" in stublog, "the container must be stopped"
    assert "curl -d " not in stublog, (
        "the smoke request must NOT be sent:\n" + stublog[-1500:]
    )


@pytest.mark.parametrize("lp", LAUNCHERS)
def test_pre_pass_post_pass_sends_smoke(lp, tmp_path):
    """Review round 35, P1: pre PASS + post PASS -> the smoke request IS sent
    and the launch succeeds (the happy path must still work)."""
    env, log, cli_env_log = _install_stubs(tmp_path)
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_LOGGING_STUB)
    curl.chmod(0o755)
    r = _run(lp, env, {"API_KEY": "secret", "IMAGE": "x@sha256:" + "a" * 64,
                       "ALLOW_RUN": "1", "KV_PERSIST": "1",
                       "GATE_PRE_RC": "0", "GATE_RC": "0"})
    combined = r.stdout + r.stderr
    assert r.returncode == 0, (
        f"the happy path must succeed, got {r.returncode}:\n{combined[-1500:]}"
    )
    stublog = log.read_text() if log.exists() else ""
    assert "identity gate (post-launch)" in combined, (
        "the post-launch gate must run on the happy path too:\n" + combined[-1500:]
    )
    assert "identity verified" in combined or "cache attached and trusted" in combined, (
        "the verified verdict must be reported:\n" + combined[-1500:]
    )
    assert "curl -d " in stublog, (
        "the smoke request MUST be sent when both gates pass:\n" + stublog[-1500:]
    )
