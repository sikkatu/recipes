"""Regression tests for bugs found in review. Each one failed before its fix.

  * hotsplit._Dyn.plan(): the consecutive-planner-failure counter was reset BEFORE the work, so the
    'N failures in a row are fatal' guard could never fire (behavioural test: needs no torch/GPU)
  * hotsplit_balance: histograms keyed by real MoE layer ids (e.g. 1..69) crashed with KeyError;
    per-token traffic was reported per layer
  * launchers: IPv6 loopback ([::1] / ::1) broke the readiness probe or the final smoke call
  * check_recipe: credential scan skipped drafts; provenance paths could escape results/<run-id>/
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from test_launcher_behaviour import run_launcher, stub_env  # noqa: F401  (stub_env is a fixture)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "recipes" / "rtx-pro-6000"
MIMO_PATCHES = BASE / "mimo-v2.6-pro-hotsplit" / "patches"


# ------------------------------------------------------------------ planner failure guard (no torch needed)
def _load_hotsplit_with_stubs(monkeypatch):
    """Import hotsplit.py under a private name with torch/vllm stubbed for the duration of one test."""
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.device = lambda *a, **k: None
    torch.nn = types.SimpleNamespace(Module=object, Parameter=object)

    class Runner:
        def execute_model(self, *a, **k):
            return "ok"

    gm = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    gm.GPUModelRunner = Runner
    for name, mod in (("torch", torch), ("vllm", types.ModuleType("vllm")),
                      ("vllm.v1", types.ModuleType("vllm.v1")), ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")),
                      ("vllm.v1.worker.gpu_model_runner", gm)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.syspath_prepend(str(MIMO_PATCHES))
    spec = importlib.util.spec_from_file_location("hotsplit_under_test", MIMO_PATCHES / "hotsplit.py")
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    return hs, Runner


class _Buf:
    def __init__(self, a):
        self.a = a

    def to(self, *_a, **_k):
        return self

    def numpy(self):
        return self.a


def _make_dyn(hs, bad: bool):
    E = 8
    dyn = object.__new__(hs._Dyn)
    dyn.every, dyn.max_swaps, dyn.decay, dyn.margin = 1, 2, 0.5, 1.25
    dyn.max_plan_errors, dyn.plan_ok, dyn.plan_errors, dyn.steps, dyn.prev = 3, 0, 0, 0, None
    dyn.crow, dyn.prior = {}, {}
    dyn.score = {0: np.zeros(E + 3 if bad else E)}      # wrong shape -> decay_scores raises inside the loop
    dyn.layers = {0: hs._DynLayer(0, [0, 1], *([None] * 8))}
    hs._LIVE = types.SimpleNamespace(buf=_Buf(np.zeros((1, E), dtype=np.int64)), layer_idx=[0])
    return dyn


def test_repeated_planner_failures_become_fatal(monkeypatch):
    hs, Runner = _load_hotsplit_with_stubs(monkeypatch)
    dyn = _make_dyn(hs, bad=True)
    assert hs._install_step_hook(dyn)
    r = Runner()
    r.execute_model()                                    # step 1: baseline snapshot only
    with pytest.raises(RuntimeError, match="planner failed"):
        for _ in range(10):
            r.execute_model()
    assert dyn.plan_errors == dyn.max_plan_errors


def test_good_pass_resets_planner_error_count(monkeypatch):
    hs, Runner = _load_hotsplit_with_stubs(monkeypatch)
    dyn = _make_dyn(hs, bad=False)
    dyn.plan_errors = 2
    assert hs._install_step_hook(dyn)
    r = Runner()
    r.execute_model()
    r.execute_model()
    assert dyn.plan_ok == 1 and dyn.plan_errors == 0


# ------------------------------------------------------------------ balance calculator
@pytest.mark.parametrize("first_layer", [0, 1, 3])
def test_balance_accepts_real_layer_ids(tmp_path, first_layer):
    layers, experts = 6, 16
    geom = {"layers": layers, "experts": experts, "topk": 2, "cell_mib_full": 4.0, "tp": 4}
    (tmp_path / "g.json").write_text(json.dumps(geom))
    rng = np.random.default_rng(0)
    hist = {str(first_layer + i): {"counts": rng.integers(1, 50, experts).tolist()} for i in range(layers)}
    (tmp_path / "h.json").write_text(json.dumps(hist))
    out = subprocess.run([sys.executable, str(BASE / "glm-5.3-hotsplit" / "patches" / "hotsplit_balance.py"),
                          str(tmp_path / "g.json"), "--counts", str(tmp_path / "h.json"), "--gib", "0.01", "--json"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["hot_cells"] > 0
    assert r["decode_mib_per_token_rank_all_cold"] == pytest.approx(layers * r["decode_mib_per_layer_token_rank"], rel=0.02)
    assert 0 < r["cold_mib_per_token_rank"] < r["decode_mib_per_token_rank_all_cold"]
    assert r["pcie_ceiling_tok_s"] > 0


def test_balance_all_hot_has_no_pcie_bound(tmp_path):
    (tmp_path / "g.json").write_text(json.dumps({"layers": 2, "experts": 4, "topk": 1, "cell_mib_full": 4.0, "tp": 4}))
    out = subprocess.run([sys.executable, str(BASE / "glm-5.3-hotsplit" / "patches" / "hotsplit_balance.py"),
                          str(tmp_path / "g.json"), "--gib", "1", "--json"], capture_output=True, text=True)
    r = json.loads(out.stdout)
    assert r["cold_mib_per_token_rank"] == 0 and r["pcie_ceiling_tok_s"] is None


# ------------------------------------------------------------------ check_recipe: drafts + path escapes
def test_metric_raw_cannot_escape_results_dir(tmp_path):
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "recipe.schema.json").write_text((ROOT / "schemas" / "recipe.schema.json").read_text())
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_recipe import check_recipe  # noqa: E402
    import yaml
    d = tmp_path / "recipes" / "hw" / "r1"
    d.mkdir(parents=True)
    (d / "README.md").write_text("## What this runs\n## Hardware\n## Software\n## Launch\n## Verify\n## Results\n## Known limits\n## Rollback\n")
    (d / "recipe.yaml").write_text(yaml.safe_dump({
        "id": "hw/r1", "name": "t", "status": "draft",
        "metrics": [{"name": "decode", "value": 1.0, "unit": "tok/s", "workload": "c1",
                     "provenance": {"run_id": "r1", "method": "m", "raw": "../../etc/hostname"}}],
    }))
    errs = check_recipe(d / "recipe.yaml", root=tmp_path)
    assert any("escapes results/r1/" in e for e in errs), errs


def test_draft_recipe_dir_is_scanned_for_credentials(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_recipe import check_draft_contract  # noqa: E402
    d = tmp_path / "recipes" / "hw" / "draft-one"
    d.mkdir(parents=True)
    (d / "README.md").write_text("# Draft\n\nAPI_KEY=sk-FAKEFAKEFAKEFAKEFAKEFAKEFAKE0000\n")
    errs = check_draft_contract(tmp_path)
    assert any("possible credential" in e and "README.md" in e for e in errs), errs
    (d / "README.md").write_text("# Draft\n\nAPI_KEY=<your key>\n")
    assert check_draft_contract(tmp_path) == []


# ------------------------------------------------------------------ launchers: IPv6 loopback
# The stub curl answers /v1/models generically but must satisfy the launch smoke test:
# a chat completion answer that contains TP4_OK (the launcher now exits 1 otherwise).
CURL_OK = (
    '#!/usr/bin/env bash\n'
    'echo "curl $*" >> "$STUB_LOG"\n'
    'case "$*" in\n'
    '  *chat/completions*) echo \'{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}\' ;;\n'
    '  *) echo \'{"ok":1}\' ;;\n'
    'esac\n'
    'exit 0\n'
)
LAUNCHERS = [
    BASE / "deepseek-v4.1-flash-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "glm-5.3-flash-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "glm-5.3-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-uva.sh",
    BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh",
]


@pytest.mark.parametrize("lp", LAUNCHERS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_smoke_failure_is_a_failed_launch(lp, stub_env):
    """Smoke test without TP4_OK in the answer must exit 1 (once it exited 0 and automation
    could read a garbage-producing run as a successful launch)."""
    env, log = stub_env
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_OK.replace('echo \'{"choices":[{"message":{"content":"TP4_OK"},"finish_reason":"stop"}]}\'',
                                   'echo \'{"choices":[{"message":{"content":"garbage output"},"finish_reason":"stop"}]}\''))
    r = run_launcher(lp, env, {"BIND": "127.0.0.1", "ALLOW_NO_AUTH": "1", "API_KEY": "",
                               "IMAGE": "x@sha256:" + "a" * 64})
    assert r.returncode == 1, f"smoke without TP4_OK must fail the launch, got rc={r.returncode}\n{r.stdout}{r.stderr}"
    assert "smoke answer does NOT contain TP4_OK" in r.stdout + r.stderr


@pytest.mark.parametrize("lp", LAUNCHERS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
@pytest.mark.parametrize("bind,host_arg,url_host", [("::1", "--host ::1", "[::1]"), ("[::1]", "--host ::1", "[::1]"),
                                                    ("127.0.0.1", "--host 127.0.0.1", "127.0.0.1")])
def test_loopback_bind_forms_work_end_to_end(lp, bind, host_arg, url_host, stub_env):
    env, log = stub_env
    curl = Path(env["PATH"].split(":")[0]) / "curl"
    curl.write_text(CURL_OK)
    r = run_launcher(lp, env, {"BIND": bind, "ALLOW_NO_AUTH": "1", "API_KEY": "",
                               "IMAGE": "x@sha256:" + "a" * 64})
    text = log.read_text()
    assert r.returncode == 0, r.stdout + r.stderr
    assert host_arg + " " in text, "server must get the bare address"
    urls = [w for line in text.splitlines() for w in line.split() if w.startswith("http://")]
    assert urls and all(u.startswith(f"http://{url_host}:") for u in urls), urls
