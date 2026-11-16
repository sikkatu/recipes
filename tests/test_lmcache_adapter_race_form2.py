"""Round 90c P0-2: race guard covers BOTH race forms on the REAL method bodies.

Form 1 (retrieve->False): guard filters the late finished_sending.
Form 2 (client abort mid-retrieve, LMCache #4185): the adapter reports
finished_sending FIRST, then -- when the in-flight load completes --
finished_recving for the SAME id; the scheduler freed the request on
finished_sending, so the late finished_recving trips the assert.
The guard snapshots `_returned_finished` (sent_before) before every
get_finished call and drops finished_recving ids seen there.

The simulator builds LMCacheMPWorkerAdapter from the REAL method bodies of
the LMCache v0.5.6 source (ast.get_source_segment over
lmcache/integration/vllm/vllm_multi_process_adapter.py): get_finished,
get_failed_request_ids, _process_finished_stores,
_update_and_get_finished_store. Without the source the tests SKIP (per spec:
never pretend-pass).

MUTATIONs caught:
  * removing the form-2 filter -> the abort test FAILS;
  * filtering finished_recving without the sent_before check -> the
    normal-order (recv-then-send) test FAILS;
  * removing the form-1 filter -> the form-1 test FAILS.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

REC = Path(__file__).resolve().parent.parent
PATCHES = REC / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "patches"
GUARD = "lmcache_mp_adapter_race"

# candidate locations of the real LMCache v0.5.6 source (reviewer stand has it
# as a checkout; the coder's workspace has the released tarball tree)
SOURCE_CANDIDATES = [
    Path("/home/work/lmcache-src/lmcache/integration/vllm/vllm_multi_process_adapter.py"),
    Path.home() / "LMCache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
    Path("/home/review/lmcache-src/lmcache/integration/vllm/vllm_multi_process_adapter.py"),
]

METHODS = ["get_finished", "get_failed_request_ids",
           "_process_finished_stores", "_update_and_get_finished_store"]


def _find_source() -> Path | None:
    for p in SOURCE_CANDIDATES:
        if p.is_file():
            return p
    return None


def _build_real_adapter_cls(src: Path):
    """Assemble a class carrying the REAL method bodies from the v0.5.6
    source (ast.get_source_segment) plus a minimal __init__ with the state
    attributes the bodies touch."""
    tree = ast.parse(src.read_text())
    segs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            segs[node.name] = ast.get_source_segment(src.read_text(), node)
    missing = [m for m in METHODS if m not in segs]
    if missing:
        pytest.skip(f"SKIP: real LMCache v0.5.6 source found but methods missing: {missing}")

    init = '''
    def __init__(self):
        self._failed_request_ids = set()
        self.error_block_ids = set()
        self.finished_stores = set()
        self.previously_finished = set()
        self._returned_finished = set()
        self.store_futures = {}
        self.retrieve_futures = {}
        self.store_events = {}
        self.retrieve_events = {}
        self._dropped_retrieves = set()
        self._pending_store_kv_events = {}
        self.model_name = "m"
        self.world_size = 1
        self.worker_id = 0
        self.dispatcher = None
        self.is_healthy = True
'''

    def _indent(seg: str) -> str:
        """ast.get_source_segment hands the method body starting at col 0 --
        inside a class body a col-0 `def` CLOSES the class (the extracted
        members silently became module-level functions). Normalize: dedent,
        then indent every line 4 spaces so the methods are class members."""
        import textwrap
        return textwrap.indent(textwrap.dedent(seg), "    ")

    body = init + "\n".join(_indent(segs[m]) for m in METHODS)
    # the real bodies call logger.error -- provide inert stubs in the exec
    # globals so the simulator exercises the REAL logic, not NameErrors.
    class _Logger:
        def error(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def info(self, *a, **k):
            pass
        def debug(self, *a, **k):
            pass
    ns: dict = {"logger": _Logger(), "dispatch": lambda *a, **k: None}
    exec(compile(ast.parse(f"class RealWorkerAdapter:\n{body}\n"), "<sim>", "exec"), ns)
    cls = ns["RealWorkerAdapter"]
    missing_after = [m for m in METHODS if not hasattr(cls, m)]
    assert not missing_after, f"simulator lost methods: {missing_after}"
    return cls


@pytest.fixture(scope="module")
def real_cls():
    src = _find_source()
    if src is None:
        pytest.skip("SKIP: real LMCache v0.5.6 source not available on this host -- "
                    "form-2 simulation needs the genuine method bodies")
    return _build_real_adapter_cls(src)


@pytest.fixture()
def guard_mod(monkeypatch, real_cls):
    monkeypatch.syspath_prepend(str(PATCHES))
    pkg_lmcache = types.ModuleType("lmcache")
    pkg_int = types.ModuleType("lmcache.integration")
    pkg_vllm = types.ModuleType("lmcache.integration.vllm")
    adapter_mod = types.ModuleType("lmcache.integration.vllm.vllm_multi_process_adapter")
    adapter_mod.LMCacheMPWorkerAdapter = real_cls
    pkg_vllm.vllm_multi_process_adapter = adapter_mod
    pkg_int.vllm = pkg_vllm
    pkg_lmcache.integration = pkg_int
    monkeypatch.setitem(sys.modules, "lmcache", pkg_lmcache)
    monkeypatch.setitem(sys.modules, "lmcache.integration", pkg_int)
    monkeypatch.setitem(sys.modules, "lmcache.integration.vllm", pkg_vllm)
    monkeypatch.setitem(
        sys.modules,
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        adapter_mod,
    )
    sys.modules.pop(GUARD, None)
    mod = importlib.import_module(GUARD)
    mod.install()
    yield mod, real_cls
    for m in ("get_finished", "get_failed_request_ids"):
        w = real_cls.__dict__.get(m)
        if w is not None and getattr(w, "__wrapped__", None) is not None:
            setattr(real_cls, m, w.__wrapped__)
    if hasattr(real_cls, "_kvp_race_guard"):
        del real_cls._kvp_race_guard
    sys.modules.pop(GUARD, None)


def _adapter(real_cls):
    a = real_cls.__new__(real_cls)   # skip upstream __init__ deps
    a._failed_request_ids = set()
    a.error_block_ids = set()
    a.finished_stores = set()
    a.previously_finished = set()
    a._returned_finished = set()
    a.store_futures = {}
    a.retrieve_futures = {}
    a.store_events = {}
    a.retrieve_events = {}
    a._dropped_retrieves = set()
    a._pending_store_kv_events = {}
    a.model_name = "m"
    a.world_size = 1
    a.worker_id = 0
    a.dispatcher = None
    a.is_healthy = True

    class _Telemetry:
        def on_request_store_finished(self, request_ids_set=None, model_name=None,
                                      world_size=None, kv_rank=None):
            pass
    a.request_telemetry = _Telemetry()
    return a


class _Future:
    """Stand-in for MessagingFuture: pending until .finish(outcome) is
    called -- .query() reports readiness, .result() the outcome (False for a
    failed retrieve)."""
    def __init__(self):
        self._ready = False
        self._outcome = None
    def finish(self, outcome):
        self._ready = True
        self._outcome = outcome
    def query(self):
        return self._ready
    def result(self, timeout=0):
        return self._outcome


def test_form2_abort_drops_late_finished_recving(guard_mod):
    """Form 2 (r90c P0-2, the reviewed LMCache #4185 shape):
      call1: engine reports X finished WHILE its retrieve is still in flight
             -> get_finished({X}) returns ({X}, set()) -- X goes out as
             finished_sending and the scheduler FREES it.
      call2: the retrieve completes (fails) -> get_finished(set()) would
             return (set(), {X}) -- a late finished_recving for a freed
             request -> scheduler assert. The guard MUST drop X here.
    MUTATION caught: removing the sent_before filter (form-2 filter) -> the
    assert-shaped sequence (set(), {X}) escapes -> this test FAILS."""
    mod, real_cls = guard_mod
    a = _adapter(real_cls)
    # call 1: retrieve IN FLIGHT (future not ready), X finished by engine
    fut = _Future()
    a.retrieve_futures["X"] = (fut, set())
    sends, recvs = a.get_finished({"X"})
    assert sends == {"X"}, f"X must go out as finished_sending first: {sends}"
    assert not recvs, f"no recvs yet -- retrieve still in flight: {recvs}"
    # call 2: the in-flight retrieve completes (fails) -> upstream would
    # report X again as finished_recving -- exactly the abort shape
    fut.finish(False)
    sends2, recvs2 = a.get_finished(set())
    assert sends2 == set()
    assert "X" not in recvs2, (
        "form 2: late finished_recving for an already-sent request must be "
        f"dropped (scheduler freed X on finished_sending); got {recvs2}")


def test_form2_normal_order_recv_then_send_untouched(guard_mod):
    """Normal order (recv first, send later) must NOT be filtered -- the
    request is alive when finished_recving arrives; only the ABORT order
    (send already seen) is guarded. MUTATION caught: filtering
    finished_recving without the sent_before check -> this test FAILS."""
    mod, real_cls = guard_mod
    a = _adapter(real_cls)
    # call 1: retrieve completes (fails) -> X goes out as finished_recving;
    # the engine has NOT finished it yet -> no finished_sending for X
    fut = _Future()
    fut.finish(False)
    a.retrieve_futures["X"] = (fut, set())
    sends, recvs = a.get_finished(set())
    assert "X" in recvs, f"normal order: finished_recving must pass: {recvs}"
    assert not sends
    # call 2: engine finishes X -> finished_sending; that is the NORMAL end
    # of a failed-load request under the fail policy (form-1 ids are in the
    # persistent set and get filtered -- X here was already reported, the
    # form-1 guard's one-shot drop applies; assert only the recv side here)
    sends2, recvs2 = a.get_finished({"X"})
    assert not recvs2, f"no duplicate recvs expected: {recvs2}"


def test_form1_still_filters_late_finished_sending(guard_mod):
    """Form 1 (r90 C2): retrieve -> False registers X in the failed-retrieve
    set; a LATER call reports X in finished_sending -> filtered.
    MUTATION caught: removing the form-1 persistent-set filter."""
    mod, real_cls = guard_mod
    a = _adapter(real_cls)
    fut = _Future()
    fut.finish(False)
    a.retrieve_futures["X"] = (fut, set())
    sends1, recvs1 = a.get_finished(set())
    assert "X" in recvs1
    # the guard recorded X from the delta of _failed_request_ids; the
    # connector may also have drained it -- either way a late send is caught
    sends2, _ = a.get_finished({"X"})
    assert "X" not in sends2, (
        f"form 1: finished_sending for a failed-retrieve id must be dropped: {sends2}")