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
import re
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


# --- Round 72 (review step-2 item 5): FULL docker-run parity with the
# persistent connector attached on BOTH sides. The reviewer measured the
# permitted differences by hand; they are pinned here as an EXPLICIT
# allowlist, so any UNEXPECTED drift (a flag silently dropped on one side)
# fails the test.

# Tokens that legitimately exist on ONE side only. Each entry is a decision:
ALLOWED_ONLY_IN_RECIPE = {
    # ONE container-name source: each path names its own engine.
    "mimo-pro-tp4-hotsplit",
    # The recipe keeps the legacy /model mount alongside the namespace mount
    # (smoke tooling historically points at /model); the engine itself reads
    # --model /models/<ns> on BOTH sides. Harmless duplicate, documented.
    "<HOST>:/model:ro",
    # Root mode: the recipe runs the engine as root, its HF cache lives under
    # /root (the kv launcher's same-UID mode uses /tmp caches instead).
    "<HOST>:/root/.cache/huggingface",
}
ALLOWED_ONLY_IN_KV = {
    "mimo-pro-tp4-kvp",
    # same-UID mode (KVP_SAME_UID=1, the kv launcher default): --user and the
    # tmp-path caches. The recipe runs the engine as root (documented
    # difference between the two launchers).
    "--user",
    "<HOST>:/tmp/hf-cache",
    "HOTSPLIT_DYN_LOG=1",
    "HOTSPLIT_STRICT_HOOK=0",
    "HOTSPLIT_NUMA_STRICT=0",
}
# Flag names whose VALUE may differ (root home vs same-UID tmp caches):
ALLOWED_VALUE_FLAGS = {
    "HF_HOME=", "XDG_CACHE_HOME=", "TRITON_CACHE_DIR=",
    "HOME=", "USER=", "LOGNAME=",
}
# Flags that consume a separate VALUE token; their values are not compared.
VALUE_FLAGS = {"--user"}


def _norm_tokens(line: str) -> list:
    """Normalize machine-specific tokens, return the token list.
    Volume mounts (host:container) are compared by their CONTAINER side -- the
    host path is check-out/pytest specific and normalized to <HOST>."""
    line = line.replace(str(KV_REPO) + "/live", "<LIVE>")
    line = re.sub(r"(/tmp/[^:\s]+)(?=/|:)", "<TMP>", line)
    out = []
    prev_value_flag = False
    for t in re.split(r"\s+", line.strip()):
        if not t:
            continue
        if prev_value_flag:
            prev_value_flag = False
            continue
        if t in VALUE_FLAGS:
            prev_value_flag = True
            out.append(t)
            continue
        if t.startswith(("/", "<")) and ":/" in t:
            t = "<HOST>:" + t.split(":", 1)[1]
        out.append(t)
    return out


def _is_value_flag(tok: str) -> bool:
    return any(tok.startswith(f) for f in ALLOWED_VALUE_FLAGS)


def test_full_docker_run_parity_with_persist(stub_env):
    """Review step-2 item 5: compare the WHOLE `docker run` of both launchers
    with the persistent connector attached (KV_PERSIST=1 / persist=true).
    Differences outside the EXPLICIT allowlist fail the test."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log = stub_env

    def run(script, extra):
        if log.exists():
            log.unlink()
        e = {**env, **extra}
        subprocess.run(["bash", str(script)], env=e, capture_output=True,
                       text=True, timeout=180)
        return _docker_run_line(log)

    common = {
        "API_KEY": "secret123", "IMAGE": "x@sha256:" + "a" * 64, "ALLOW_RUN": "1",
        "KVP_SAME_UID": "1", "KVP_SKIP_GPU_CHECK": "1",
        "KVP_SKIP_HOTSPLIT_CHECK": "1", "KVP_SKIP_IPC_TEST": "1",
        "RECIPE_DIR": str(RECIPE),
        "KV_CACHE_DTYPE": "bfloat16",
        "KVP_REPO": str(KV_REPO),
        # Round 73 (review): HOTSPLIT_HOT_GIB must be COMPARED on the
        # delegated path, not allow-listed. start-all.sh exports the shared
        # default (_kvp_hotsplit_hot_gib_default=30) to BOTH paths -- this
        # test reproduces exactly that, so both docker runs must carry the
        # SAME value and the equality assertion below verifies it.
        "HOTSPLIT_HOT_GIB": "30",
        # ONE namespace on both sides (the parity hook): the recipe honours
        # KV_PERSIST_NAMESPACE, the kv launcher KVP_MODEL_NAME -- the same
        # escape hatch the delegation uses.
        "KV_PERSIST_NAMESPACE": "parity-ns",
        # Round 73 (review security): the namespace override needs EXPLICIT
        # consent -- the test passes it (that is what the hook is for).
        "KVP_ALLOW_NAMESPACE_OVERRIDE": "1",
        "KVP_MODEL_NAME": "parity-ns",
    }
    # The MANDATORY identity gate runs on both sides (pre-launch) -- it needs
    # the REQUIRED identity pins in configs/mimo-v2.6-pro/identity.env.local.
    # Fill them temporarily and RESTORE the operator's real file afterwards
    # (the same contract as tests/unit/test_identity_gate.py).
    import re as _re
    _cfg = KV_REPO / "configs" / "mimo-v2.6-pro"
    local = _cfg / "identity.env.local"
    backup = local.read_text() if local.exists() else None
    base = (_cfg / "identity.env").read_text()
    for name, value in (("KVP_MODEL_REVISION", "0" * 40),
                        ("KVP_TOKENIZER_REVISION", "0" * 40),
                        ("KVP_CHAT_TEMPLATE_DIGEST", "0" * 64)):
        base = _re.sub(r"^(" + name + r")=.*$", r"\1=" + value, base,
                       flags=_re.M)
    local.write_text(base)
    # The kv launcher with the connector needs a "running" cache server: the
    # pid check reads the pid file; a live PID satisfies it. The reachability
    # pre-flight needs an open socket -- a listener on the configured port.
    import socket
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(5)
    try:
        pidfile = Path(env["LIVE_DIR"]) / "lmcache-server.pid"
        pidfile.write_text(str(os.getpid()))
        persist = {"KVP_CACHE_ROOT": env["LIVE_DIR"],
                   "LMCACHE_HOST": "127.0.0.1", "LMCACHE_PORT": str(port),
                   "KV_PERSIST_HOST": "127.0.0.1", "KV_PERSIST_PORT": str(port)}
        recipe_line = run(RECIPE / "scripts" / "run-tp4-hotsplit.sh",
                          {**common, **persist, "KV_PERSIST": "1"})
        kv_line = run(KV_REPO / "scripts" / "run-vllm-with-persist.sh",
                      {**common, **persist, "KVP_PERSIST_ENABLED": "true"})
    finally:
        srv.close()
        if backup is None:
            local.unlink(missing_ok=True)
        else:
            local.write_text(backup)
    assert recipe_line, "recipe launcher (KV_PERSIST=1) did not reach docker run"
    assert kv_line, "KV launcher (persist=true) did not reach docker run"

    r_tokens = _norm_tokens(recipe_line)
    k_tokens = _norm_tokens(kv_line)

    # 1. the persist-ON contract flags must be on BOTH sides (this is the
    #    reviewer's list: the recipe used to add --block-size and
    #    --enable-prompt-tokens-details ONLY under KV_PERSIST=1, and the
    #    --kv-transfer-config is what actually attaches the cache).
    for flag in ("--kv-transfer-config", "--block-size", "--enable-prompt-tokens-details",
                 "--cap-add IPC_LOCK", "--cap-add SYS_NICE", "--compilation-config"):
        assert flag in recipe_line, (
            f"recipe docker run (persist ON) is missing {flag}")
        assert flag in kv_line, (
            f"KV docker run (persist ON) is missing {flag} (parity broken)")

    # 2. the key must never be an argv value on either side
    for side, name in ((recipe_line, "recipe"), (kv_line, "KV")):
        assert "secret123" not in side, f"{name}: key leaked into argv"
        assert "VLLM_API_KEY=" not in side.split("--env-file")[0], (
            f"{name}: key on the command line")

    # 3. MULTISET comparison (round 78: set() lost duplicate flags/mounts
    #    and ordering). Counter tracks multiplicity: if one side has --foo
    #    twice and the other once, the difference is visible. We also detect
    #    unexpected DUPLICATE flags (a flag appearing more than once outside
    #    the documented multi-use flags like --cap-add or -v).
    from collections import Counter
    MULTI_USE_FLAGS = {"--cap-add", "-v", "--mount", "-e", "--env-file",
                       "--gpus", "--shm-size", "--ulimit"}
    r_cnt, k_cnt = Counter(r_tokens), Counter(k_tokens)
    # Tokens appearing on ONE side only (outside allowlist) = drift.
    # MULTI-USE flag tokens are exempt here: their VALUES are compared
    # separately (3d flag-value multiset + 3c mount order), and a flag token
    # like '-v' legitimately appears more times on one side when a documented
    # extra mount exists (the recipe's legacy /model).
    unexpected_r = {t for t in (r_cnt - k_cnt)
                    if t not in ALLOWED_ONLY_IN_RECIPE
                    and t not in MULTI_USE_FLAGS
                    and not _is_value_flag(t)
                    and not t.startswith(("<TMP>/", "<LIVE>"))}
    unexpected_k = {t for t in (k_cnt - r_cnt)
                    if t not in ALLOWED_ONLY_IN_KV
                    and t not in MULTI_USE_FLAGS
                    and not _is_value_flag(t)
                    and not t.startswith(("<TMP>/", "<LIVE>"))}
    assert not unexpected_r, (
        f"tokens ONLY in the recipe docker run outside the allowlist: "
        f"{sorted(unexpected_r)}")
    assert not unexpected_k, (
        f"tokens ONLY in the KV docker run outside the allowlist: "
        f"{sorted(unexpected_k)}")
    # 3b. duplicate-flag detection: a non-multi-use flag appearing >1 time
    #     on one side is suspicious (e.g. accidentally doubled --model).
    for side_name, tokens in (("recipe", r_tokens), ("KV", k_tokens)):
        cnt = Counter(tokens)
        for flag, n in cnt.items():
            if n > 1 and flag.startswith("--") and flag not in MULTI_USE_FLAGS:
                # Check it is not a VALUE token after a multi-use flag
                # (e.g. "-e HOTSPLIT_X=1" -- the value is not a flag).
                pass  # flag tokens are --foo; values don't start with --
                raise AssertionError(
                    f"{side_name}: flag {flag!r} appears {n} times "
                    f"(duplicate flags indicate a docker-run construction bug)")
    # 3d. multi-use flag VALUES must match with multiplicity (the old set()
    #     and a bare flag-token check would both miss a doubled -e or a
    #     dropped mount). Values outside the documented differences fail.
    def _flag_values(tokens, flag):
        return [tokens[i + 1] for i, t in enumerate(tokens)
                if t == flag and i + 1 < len(tokens)]
    def _nv(vs):
        out = []
        for v in vs:
            if v.endswith("/.engine-env"):
                # key env-file: <LIVE>/.engine-env (kv) and <TMP>/.engine-env
                # (recipe, LIVE_DIR under /tmp in tests) are the SAME artifact
                v = "<ENVFILE>"
            if ":" in v and v.startswith(("/", "<")):
                out.append("<HOST>:" + v.split(":", 1)[1])
            else:
                out.append(v)
        return out
    for flag in ("-v", "-e", "--env-file", "--cap-add"):
        # _is_value_flag: env values whose VALUE may legitimately differ
        # (root HF cache vs same-UID tmp caches) -- they are compared for
        # PRESENCE by name elsewhere, the value is an allowed difference.
        r_vals = [v for v in _flag_values(r_tokens, flag)
                  if v not in ALLOWED_ONLY_IN_RECIPE and not _is_value_flag(v)]
        k_vals = [v for v in _flag_values(k_tokens, flag)
                  if v not in ALLOWED_ONLY_IN_KV and not _is_value_flag(v)]
        assert Counter(_nv(r_vals)) == Counter(_nv(k_vals)), (
            f"{flag} values must match with multiplicity: "
            f"recipe={sorted(_nv(r_vals))} KV={sorted(_nv(k_vals))}")

    # 3c. mounts compared as a MULTISET of container-side targets (round 78).
    #     The ORDER of docker-run flags is historically different between the
    #     two launchers and semantically irrelevant to docker -- requiring
    #     positional equality would be a false red. What matters is the SET of
    #     mounts with multiplicity (a dropped/duplicated mount is a real
    #     drift) and that no mount target appears twice (overlap bug).
    def _mount_targets(tokens):
        out = []
        for i, t in enumerate(tokens):
            if t == "-v" and i + 1 < len(tokens):
                target = tokens[i + 1].split(":", 2)[1] if ":" in tokens[i + 1] else tokens[i + 1]
                out.append(target)
        return out
    r_mnts, k_mnts = _mount_targets(r_tokens), _mount_targets(k_tokens)
    # Documented mount differences (legacy /model, root vs same-UID HF cache)
    DOC_MNTS = {"/model", "/root/.cache/huggingface", "/tmp/hf-cache"}
    r_m = [m for m in r_mnts if m not in DOC_MNTS]
    k_m = [m for m in k_mnts if m not in DOC_MNTS]
    assert Counter(r_m) == Counter(k_m), (
        f"mount targets must match with multiplicity: recipe={sorted(r_m)} "
        f"KV={sorted(k_m)}")
    for side_name, mnts in (("recipe", r_mnts), ("KV", k_mnts)):
        cnt = Counter(mnts)
        dups = [t for t, n in cnt.items() if n > 1]
        assert not dups, (
            f"{side_name}: mount target(s) duplicated in the docker run: {dups}")

    # 4. HOTSPLIT_HOT_GIB must be EQUAL on both sides (review: it was
    #    allow-listed as a permitted difference -- the default could drift).
    #    The recipe launcher reads the value from the environment/tp4.env;
    #    the kv launcher from the shared env.sh helper. With no tp4.env and
    #    no explicit export the resolved value must match.
    def _hotsplit_val(tokens, name):
        for i, t in enumerate(tokens):
            if t == "-e" and i + 1 < len(tokens) and tokens[i + 1].startswith(name + "="):
                return tokens[i + 1].split("=", 1)[1]
        return None
    r_hot = _hotsplit_val(r_tokens, "HOTSPLIT_HOT_GIB")
    k_hot = _hotsplit_val(k_tokens, "HOTSPLIT_HOT_GIB")
    # The kv side ALWAYS passes it (the shared default); the recipe side
    # passes it when set. Both must resolve to the SAME value -- and that
    # value is the documented 30.
    assert k_hot == "30", f"kv side hot budget must be the shared default 30, got {k_hot}"
    if r_hot is not None:
        assert r_hot == k_hot, (
            f"HOTSPLIT_HOT_GIB must be EQUAL on both paths: recipe={r_hot} kv={k_hot}")

    # 5. the namespace mount must be the SAME on both sides (the parity hook)
    def _model_target(tokens):
        for i, t in enumerate(tokens):
            if t == "--model" and i + 1 < len(tokens):
                return tokens[i + 1]
        return None
    r_m, k_m = _model_target(r_tokens), _model_target(k_tokens)
    assert r_m and k_m and r_m == k_m, (
        f"the model mount path (namespace) must match: {r_m} vs {k_m}")
