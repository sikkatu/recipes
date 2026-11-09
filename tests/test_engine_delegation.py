"""Round 70: delegation tests -- KVP_ENGINE=recipe (review step-2 checklist).

Every checklist item, verified without hardware (stub docker/curl):

1. WITHOUT the flag nothing changes -- the default path launches the KV
   launcher, not the recipe one (the byte-level contract is the snapshot
   test in test_activation_negative.py).
2. WITH KVP_ENGINE=recipe: the recipe launcher runs, its activation-failure
   exit code REACHES the caller (start-all foreground), and the recorded
   container name is the one stop-engine.sh uses.
3. ONE key source: the shared key is exported in the environment; it must
   never appear in argv (ps).
4. KV_CACHE_DTYPE: the delegation pins bfloat16 (the recipe default is
   auto) -- the difference is SHOWN here, never normalized away. The
   substitution itself is verified BEHAVIOURALLY (round 72, review: a
   comment check is weak): the real resolve_effective_identity is run
   against a fake config.json (torch_dtype: bfloat16) -- `auto` and the
   pinned `bfloat16` resolve to the SAME value and the SAME cache
   namespace (real CacheIdentity), `fp8` to a different one. The pin
   cannot move the warmed cache's namespace.
"""
from __future__ import annotations
import os
import stat
import subprocess
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parent.parent
KV_REPO = next(
    (c for c in (
        RECIPES / "kv-persistence",
        Path.home() / "kv-persistence",
        RECIPES.parent / "kv-persistence",
    ) if c.is_dir()),
    None,
)
RECIPE = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
SKIP = "kv-persistence checkout not found (delegation tests need it)"

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
      logs) printf '%s' "${STUB_DOCKER_LOGS:-}" ;;
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
    for d in ("model", "home", "live"):
        (tmp_path / d).mkdir()
    home = tmp_path / "home"
    (home / ".config" / "kvp").mkdir(parents=True)
    key_file = home / ".config" / "kvp" / "api_key"
    key_file.write_text("sharedkey123\n")
    key_file.chmod(0o600)
    # the cache step of start-all needs the real lmcache CLI -- a running-PID
    # stub makes it skip; these tests exercise the ENGINE step only
    (tmp_path / "live" / "lmcache-server.pid").write_text(str(os.getpid()))
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["MODEL_DIR"] = str(tmp_path / "model")
    env["HF_CACHE"] = str(tmp_path / "hf")
    env["LIVE_DIR"] = str(tmp_path / "live")
    env["HOME"] = str(home)
    env["KVP_API_KEY_FILE"] = str(key_file)
    env["KVP_CACHE_ROOT"] = str(tmp_path / "live")
    env["WAIT_TRIES"] = "1"
    env["WAIT_SLEEP"] = "0"
    env["KVP_SKIP_HOTSPLIT_CHECK"] = "1"
    # Round 77: the delegated branch now runs the shared pre-flights (mp_ping +
    # CUDA IPC test) BEFORE docker run -- they need real hardware (GPU, lmcache
    # venv). These tests assert the DELEGATION wiring, so opt out exactly like
    # the hotsplit check above.
    env["KVP_SKIP_IPC_TEST"] = "1"
    # A FAKE kv-persistence checkout with a stub identity gate -- the same
    # device as test_kv_persist_behaviour (round 34). The mandatory gate needs
    # a real identity (hardware/setup territory); the DELEGATION wiring is
    # what these tests assert. The stub gate logs and passes.
    kvp = tmp_path / "kv-persistence"
    (kvp / "implementation" / "kvpersist").mkdir(parents=True)
    (kvp / "implementation" / "kvpersist" / "cli.py").write_text("# stub\n")
    (kvp / "KVP_COMPAT_VERSION").write_text("1\n")
    (kvp / "scripts").mkdir(exist_ok=True)
    gate = kvp / "scripts" / "verify-engine-identity.sh"
    gate.write_text(
        '#!/usr/bin/env bash\n'
        'echo "$0 $*" >> "$STUB_LOG"\n'
        'case "$*" in\n'
        '  *--check-config-only*) exit "${GATE_PRE_RC:-${GATE_RC:-0}}" ;;\n'
        'esac\n'
        'exit "${GATE_RC:-0}"\n'
    )
    gate.chmod(gate.stat().st_mode | stat.S_IEXEC)
    env["KVP_REPO"] = str(kvp)
    env["KVP_COMPAT_VERSION"] = "1"
    # the KV_PERSIST block derives the namespace via the kv identity CLI
    # (needs a real 40-hex model revision) -- the documented escape hatch
    env["KV_PERSIST_NAMESPACE"] = "test-namespace"
    return env, log, tmp_path


def _start_all(env, log, extra, marker_logs=""):
    if log.exists():
        log.unlink()
    e = {**env, "STUB_DOCKER_LOGS": marker_logs, **extra}
    return subprocess.run(
        ["bash", str(KV_REPO / "scripts" / "start-all.sh")], env=e,
        capture_output=True, text=True, timeout=180)


def _docker_run_line(log: Path):
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        if "docker run " in line and "--rm" not in line:
            return line
    return None


def test_default_engine_is_the_kv_launcher(stub_env):
    """Checklist 1: WITHOUT the flag start-all launches the kv launcher."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    r = _start_all(env, log, {"RECIPE_DIR": str(RECIPE),
                              "IMAGE": "x@sha256:" + "a" * 64,
                              # this test asserts the LAUNCHER CHOICE only --
                              # skip the cache pre-flight (needs a live server)
                              "KVP_PERSIST_ENABLED": "false"}, GOOD_MARKERS)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-800:]
    # the kv launcher's own signals: its container name in the docker run and
    # NO delegation banner (the recipe path prints "KVP_ENGINE=recipe:")
    assert "KVP_ENGINE=recipe" not in out, "the default path must NOT delegate"
    line = _docker_run_line(log)
    assert line and "--name mimo-pro-tp4-kvp" in line, line


def test_recipe_path_launches_the_recipe_launcher(stub_env):
    """WITH KVP_ENGINE=recipe the recipe launcher runs and the recorded
    container name matches its own --name (ONE name source)."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    r = _start_all(env, log, {"RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
                   "IMAGE": "x@sha256:" + "a" * 64}, GOOD_MARKERS)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-800:]
    assert "KVP_ENGINE=recipe" in out
    rec = (KV_REPO / "live" / "engine_container").read_text().strip()
    assert rec == "mimo-pro-tp4-hotsplit", rec
    line = _docker_run_line(log)
    assert line and f"--name {rec}" in line, line


def test_recipe_path_key_never_in_argv(stub_env):
    """Checklist 4: the shared key travels in the environment, never in argv."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    r = _start_all(env, log, {"RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
                   "IMAGE": "x@sha256:" + "a" * 64}, GOOD_MARKERS)
    assert log.exists(), (r.stdout + r.stderr)[-600:]
    for line in log.read_text().splitlines():
        assert "sharedkey123" not in line, f"key leaked into argv: {line[:120]}"


def test_recipe_path_exit_code_reaches_the_caller(stub_env):
    """Checklist 2: a FAILED activation in the recipe launcher must fail
    start-all with the SAME exit code -- never swallowed."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    e = {**env, "STUB_DOCKER_LOGS": "no markers here\n",
         "RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
         "IMAGE": "x@sha256:" + "a" * 64,
         "KVP_SKIP_HOTSPLIT_CHECK": "0"}
    if log.exists():
        log.unlink()
    r = subprocess.run(["bash", str(KV_REPO / "scripts" / "start-all.sh")], env=e,
                       capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    assert r.returncode == 1, (
        f"the recipe launcher's exit 1 must reach the caller, got {r.returncode}\n{out[-800:]}")
    assert "NOT VERIFIED" in out or "activation" in out, out[-800:]


def test_recipe_path_arbitrary_exit_code_through_kvp_up(stub_env):
    """Review round 78: the exit code must propagate through the FULL chain
    (kvp up -> exec -> start-all.sh -> recipe launcher), not just exit 1 via
    direct start-all.sh invocation. GATE_RC=42 makes the identity gate exit
    42; kvp up must return 42 to the caller."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    # A stub recipe dir whose launcher exits with the ARBITRARY code -- the
    # chain kvp up -> exec -> start-all.sh -> foreground bash must carry it
    # verbatim (the real launcher wraps failures in `die`=exit 1, which the
    # sibling test already covers).
    import stat as _stat
    # build the stub recipe inline (tmp_path is not in scope here):
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="r78-stub-recipe-"))
    (d / "scripts").mkdir(parents=True)
    launcher = d / "scripts" / "run-tp4-hotsplit.sh"
    launcher.write_text("#!/usr/bin/env bash\nexit 42\n")
    launcher.chmod(launcher.stat().st_mode | _stat.S_IEXEC)
    e = {**env, "STUB_DOCKER_LOGS": GOOD_MARKERS, "RECIPE_DIR": str(d),
         "KVP_ENGINE": "recipe", "IMAGE": "x@sha256:" + "a" * 64,
         "KVP_SKIP_HOTSPLIT_CHECK": "1",
         "KVP_SKIP_IPC_TEST": "1",
         "KVP_REPO": str(KV_REPO)}
    kvp_script = KV_REPO / "scripts" / "kvp"
    if not kvp_script.exists():
        pytest.skip("kvp script not found in checkout")
    if log.exists():
        log.unlink()
    try:
        r = subprocess.run(["bash", str(kvp_script), "up"], env=e,
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 42, (
            f"arbitrary exit code 42 must propagate through kvp up, "
            f"got {r.returncode}\n{(r.stdout + r.stderr)[-600:]}")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_container_env_override_wins_on_the_recipe_path(stub_env):
    """Review round 78 finding 1: CONTAINER= from the environment must name
    the launched container -- the old code overwrote it with the
    KVP_ENGINE_CONTAINER default, so `CONTAINER=my-custom KVP_ENGINE=recipe`
    launched and recorded the WRONG name."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    rec = KV_REPO / "live" / "engine_container"
    rec.unlink(missing_ok=True)
    r = _start_all(env, log, {"RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
                   "IMAGE": "x@sha256:" + "a" * 64, "CONTAINER": "my-custom"},
                   GOOD_MARKERS)
    rec_val = rec.read_text().strip() if rec.exists() else ""
    rec.unlink(missing_ok=True)
    assert rec_val == "my-custom", (
        f"CONTAINER= override must win and be recorded, got {rec_val!r}\n"
        f"{(r.stdout + r.stderr)[-600:]}")
    line = _docker_run_line(log)
    assert line and "--name my-custom" in line, (
        f"the docker run must carry the overridden name: {line}")


def test_env_api_key_replaces_the_canonical_key_file(stub_env):
    """Review round 78 finding 2: an explicit API_KEY must replace the
    canonical key file (otherwise the running server and every tool reading
    ~/.config/kvp/api_key see DIFFERENT keys -- a split-brain)."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    key_file = Path(env["KVP_API_KEY_FILE"])
    key_file.write_text("oldkey-oldkey-oldkey-oldkey-xxxx\n")
    r = _start_all(env, log, {"RECIPE_DIR": str(RECIPE), "KVP_ENGINE": "recipe",
                   "IMAGE": "x@sha256:" + "a" * 64,
                   "API_KEY": "newkey-newkey-newkey-newkey-xxxx"},
                   GOOD_MARKERS)
    canon = key_file.read_text().strip()
    assert canon == "newkey-newkey-newkey-newkey-xxxx", (
        f"the canonical key file must hold the env key, got {canon!r}")
    live = Path(env["LIVE_DIR"]) / "api_key"
    live_val = live.read_text().strip() if live.exists() else ""
    assert live_val == canon, (
        f"live/api_key must mirror the canonical key: {live_val!r} vs {canon!r}")


def test_stop_engine_reads_the_recorded_container(stub_env):
    """Checklist 3: stop-engine.sh (kvp down / stop-all.sh) stops the
    container that was ACTUALLY started -- the one recorded by start-all."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, log, _ = stub_env
    rec_file = KV_REPO / "live" / "engine_container"
    rec_file.parent.mkdir(parents=True, exist_ok=True)
    rec_file.write_text("mimo-pro-tp4-hotsplit\n")
    try:
        if log.exists():
            log.unlink()
        r = subprocess.run(["bash", str(KV_REPO / "scripts" / "stop-engine.sh")],
                           env=env, capture_output=True, text=True, timeout=60)
        out = r.stdout + r.stderr
        assert "mimo-pro-tp4-hotsplit" in out, (
            f"stop-engine must resolve the recorded container name:\n{out}")
    finally:
        rec_file.unlink(missing_ok=True)


def test_kv_cache_dtype_pin_and_identity(stub_env):
    """Checklist 5: the delegation pins bfloat16 while the recipe default is
    auto -- the difference stays VISIBLE (not normalized away), the identity
    hash accounts for the dtype (each dtype -> its own namespace), and the
    auto->bfloat16 resolution is documented in the launcher source so the pin
    cannot move the warmed cache's namespace."""
    if KV_REPO is None:
        pytest.skip(SKIP)
    env, _, _ = stub_env

    # (a) the recipe default is auto; the delegation pins bfloat16 explicitly
    recipe_text = (RECIPE / "scripts" / "run-tp4-hotsplit.sh").read_text()
    assert 'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"' in recipe_text, (
        "the recipe default is auto -- the delegation pin must be explicit")
    start_all = (KV_REPO / "scripts" / "start-all.sh").read_text()
    assert 'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"' in start_all, (
        "the delegation must pin KV_CACHE_DTYPE=bfloat16")

    # (b) the identity hash accounts for the dtype: build a real CacheIdentity
    # per dtype and compare the namespace hashes
    def ns_for(dtype: str) -> str:
        script = (
            "import sys, json, hashlib; "
            f"sys.path.insert(0, r'{KV_REPO / 'implementation'}'); "
            "from kvpersist.identity import CacheIdentity; "
            "i = CacheIdentity("
            "model_id='m', model_revision='a'*40, "
            "tokenizer_name='t', tokenizer_revision='b'*40, "
            f"kv_cache_dtype={dtype!r}, "
            "block_size=16, num_kv_heads=8, head_dim=192, tp_size=4, "
            "connector='LMCacheMPConnector', connector_version='0.5.6', "
            "max_model_len=1024, chat_template_digest='c'*64, "
            "l2_serde='zstd'); "
            "print(hashlib.sha256(json.dumps(i.canonical(), sort_keys=True)"
            ".encode()).hexdigest()[:16])"
        )
        r = subprocess.run(["python3", "-c", script],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (
            f"identity build failed for {dtype}: {r.stderr[-400:]}")
        return r.stdout.strip()

    ns_bf16 = ns_for("bfloat16")
    ns_fp16 = ns_for("float16")
    ns_fp32 = ns_for("float32")
    assert len({ns_bf16, ns_fp16, ns_fp32}) == 3, (
        "the identity hash must change with the dtype (each dtype -> its own "
        f"namespace): bf16={ns_bf16} fp16={ns_fp16} fp32={ns_fp32}")

    # (c) BEHAVIOURAL (review: a comment check is weak). Run the REAL
    # resolution block (kvpersist.cli resolve_effective_identity) against a
    # fake config.json with torch_dtype: bfloat16. `auto` and the pinned
    # `bfloat16` must resolve to the SAME value -- and produce the SAME
    # namespace via the real CacheIdentity. This is the substance of "the pin
    # cannot move the warmed cache's namespace": auto resolves to the model
    # dtype BEFORE hashing, so the pin is a no-op for a bf16 model.
    import json as _json
    import tempfile as _tf

    def resolve_dtype(env_dtype: str) -> tuple:
        """Return (resolved_kv_dtype, namespace) for the given KV_CACHE_DTYPE
        env value, using the REAL code path: resolve_effective_identity with a
        fake config.json, then a real CacheIdentity built from those values."""
        with _tf.TemporaryDirectory() as td:
            cfg_dir = Path(td)
            (cfg_dir / "config.json").write_text(_json.dumps(
                {"torch_dtype": "bfloat16", "dtype": "float32"}))
            script = (
                "import sys, json, hashlib; "
                f"sys.path.insert(0, r'{KV_REPO / 'implementation'}'); "
                "import os; "
                f"os.environ['KV_CACHE_DTYPE'] = {env_dtype!r}; "
                "os.environ.pop('KVP_KV_CACHE_DTYPE', None); "
                "from kvpersist.cli import resolve_effective_identity; "
                f"eff = resolve_effective_identity(r'{cfg_dir}'); "
                "from kvpersist.identity import CacheIdentity; "
                "i = CacheIdentity("
                "model_id='m', model_revision='a'*40, "
                "tokenizer_name='t', tokenizer_revision='b'*40, "
                "kv_cache_dtype=eff['KVP_KV_CACHE_DTYPE'], "
                "block_size=16, num_kv_heads=8, head_dim=192, tp_size=4, "
                "connector='LMCacheMPConnector', connector_version='0.5.6', "
                "max_model_len=1024, chat_template_digest='c'*64, "
                "l2_serde='zstd'); "
                "print(eff['KVP_KV_CACHE_DTYPE']); "
                "print(hashlib.sha256(json.dumps(i.canonical(), sort_keys=True)"
                ".encode()).hexdigest()[:16])"
            )
            r = subprocess.run(["python3", "-c", script],
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, (
                f"resolve_effective_identity failed for {env_dtype!r}: "
                f"{r.stderr[-500:]}")
            lines = [ln for ln in r.stdout.strip().splitlines() if ln]
            assert len(lines) == 2, (env_dtype, r.stdout, r.stderr)
            return lines[0], lines[1]

    resolved_auto, ns_auto = resolve_dtype("auto")
    resolved_pin, ns_pin = resolve_dtype("bfloat16")
    resolved_fp8, ns_fp8 = resolve_dtype("fp8")
    # torch_dtype beats dtype (bfloat16, not the dtype-key float32)
    assert resolved_auto == "bfloat16", (
        f"`auto` must resolve to the model's torch_dtype bfloat16, "
        f"got {resolved_auto!r} (the config.json sets both torch_dtype and "
        f"dtype -- torch_dtype wins)")
    # the SUBSTITUTION: auto and the pin resolve to the SAME value...
    assert resolved_pin == resolved_auto == "bfloat16", (
        f"the pinned bfloat16 must resolve to the same value as auto: "
        f"{resolved_pin!r} vs {resolved_auto!r}")
    # ...and therefore the SAME namespace (the warmed cache is NOT moved)
    assert ns_pin == ns_auto, (
        f"the pin must not move the warmed cache's namespace: "
        f"auto={ns_auto} bfloat16={ns_pin}")
    # while a genuinely different dtype lands in its own namespace
    assert ns_fp8 != ns_auto, (
        f"fp8 must land in a DIFFERENT namespace than bfloat16: "
        f"{ns_fp8} vs {ns_auto}")
    # NEGATIVE boundary (review): when the model config declares NO dtype at
    # all (no torch_dtype, no dtype), `auto` CANNOT resolve -- it stays
    # `auto` and therefore hashes to a DIFFERENT namespace than the pinned
    # bfloat16. That is a SAFE cache miss (never a wrong read), but the
    # boundary must be described: the pin is a no-op only for a model whose
    # config declares its dtype.
    def resolve_dtype_noconfig(env_dtype: str) -> tuple:
        with _tf.TemporaryDirectory() as td:
            cfg_dir = Path(td)
            (cfg_dir / "config.json").write_text(_json.dumps({"model_type": "mimo"}))
            script = (
                "import sys, json, hashlib; "
                f"sys.path.insert(0, r'{KV_REPO / 'implementation'}'); "
                "import os; "
                f"os.environ['KV_CACHE_DTYPE'] = {env_dtype!r}; "
                "os.environ.pop('KVP_KV_CACHE_DTYPE', None); "
                "from kvpersist.cli import resolve_effective_identity; "
                f"eff = resolve_effective_identity(r'{cfg_dir}'); "
                "from kvpersist.identity import CacheIdentity; "
                "i = CacheIdentity("
                "model_id='m', model_revision='a'*40, "
                "tokenizer_name='t', tokenizer_revision='b'*40, "
                "kv_cache_dtype=eff['KVP_KV_CACHE_DTYPE'], "
                "block_size=16, num_kv_heads=8, head_dim=192, tp_size=4, "
                "connector='LMCacheMPConnector', connector_version='0.5.6', "
                "max_model_len=1024, chat_template_digest='c'*64, "
                "l2_serde='zstd'); "
                "print(eff['KVP_KV_CACHE_DTYPE']); "
                "print(hashlib.sha256(json.dumps(i.canonical(), sort_keys=True)"
                ".encode()).hexdigest()[:16])"
            )
            r = subprocess.run(["python3", "-c", script],
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stderr[-400:]
            lines = [ln for ln in r.stdout.strip().splitlines() if ln]
            assert len(lines) == 2, (env_dtype, r.stdout, r.stderr)
            return lines[0], lines[1]

    resolved_nodtype, ns_nodtype = resolve_dtype_noconfig("auto")
    assert resolved_nodtype == "auto", (
        f"with no dtype in config.json, `auto` must stay `auto`, "
        f"got {resolved_nodtype!r}")
    assert ns_nodtype != ns_pin, (
        f"unresolved `auto` must NOT share the bfloat16 namespace "
        f"(the pin cannot move an UNRESOLVED cache -- safe miss): "
        f"{ns_nodtype} vs {ns_pin}")
