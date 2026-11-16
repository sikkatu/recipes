"""Round 90b C2 (fixed): LMCache race guard covers LMCacheMPWorkerAdapter.

The r90 version of the patch wrapped LMCacheMPSchedulerAdapter.get_finished --
a class that has NO such method in the real LMCache v0.5.6 (verified by AST
scan of the source; the fixture below records the actual API surface), so
install() reported SKIP and a fake test passed against a mock of a
non-existent method.

This file tests the FIXED behaviour:
  * install() targets LMCacheMPWorkerAdapter (the real owner of get_finished);
  * the guard keeps its OWN persistent failed-retrieve set (upstream drains
    _failed_request_ids through get_failed_request_ids() after every poll --
    a LATE finished_sending in a LATER call must still be filtered);
  * get_failed_request_ids is intercepted so the drained ids are copied into
    the guard before the next poll wipes the upstream set;
  * the API fixture guards against drift: a renamed method must FAIL.

MUTATIONs caught:
  * pointing install() back at LMCacheMPSchedulerAdapter (the r90 bug);
  * dropping the persistent-set logic (filter only against upstream per-call
    set) -> the inter-call test FAILS;
  * dropping the get_failed_request_ids interception -> same test FAILS.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REC = Path(__file__).resolve().parent.parent
PATCHES = REC / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "patches"
FIXTURE = REC / "tests" / "fixtures" / "lmcache-0.5.6-adapter-api.json"
GUARD = "lmcache_mp_adapter_race"


class _FakeWorker:
    """Faithful to the REAL LMCacheMPWorkerAdapter surface the guard relies
    on: get_finished, get_finished_with_lazy_offload, get_failed_request_ids
    and the DRAINABLE `_failed_request_ids` set (see the fixture)."""

    def __init__(self):
        self._failed_request_ids = set()
        self._pending_sends = set()
        self._pending_recvs = set()
        self._fail_this_call = set()   # retrieve ids that FAIL inside this call

    def get_finished(self, finished_req_ids_from_engine):
        # upstream (v0.5.6, lines ~2083/2138): a failed retrieve registers its
        # id in _failed_request_ids DURING this very call and is returned in
        # finished_retrieves; the guard's before/after snapshot must catch it.
        for rid in self._fail_this_call:
            self._failed_request_ids.add(rid)
        self._fail_this_call = set()
        return self._pending_sends & finished_req_ids_from_engine, set(self._pending_recvs)

    def get_finished_with_lazy_offload(self):
        return set(self._pending_sends), set(self._pending_recvs)

    def get_failed_request_ids(self):
        out = self._failed_request_ids
        self._failed_request_ids = set()      # upstream DRAINS here
        return out


@pytest.fixture()
def guard_mod(monkeypatch):
    monkeypatch.syspath_prepend(str(PATCHES))
    pkg_lmcache = types.ModuleType("lmcache")
    pkg_int = types.ModuleType("lmcache.integration")
    pkg_vllm = types.ModuleType("lmcache.integration.vllm")
    adapter_mod = types.ModuleType("lmcache.integration.vllm.vllm_multi_process_adapter")
    adapter_mod.LMCacheMPWorkerAdapter = _FakeWorker      # the REAL class name
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
    yield mod, _FakeWorker
    if hasattr(_FakeWorker, "_kvp_race_guard"):
        del _FakeWorker._kvp_race_guard
    for m in ("get_finished", "get_finished_with_lazy_offload", "get_failed_request_ids"):
        w = getattr(_FakeWorker, "__dict__", {}).get(m)
        if w is not None and getattr(w, "_kvp_wrapped", False):
            delattr(_FakeWorker, m)   # unwrap so the next test starts pristine
    sys.modules.pop(GUARD, None)


def test_install_wraps_worker_adapter_not_scheduler(guard_mod):
    """r90b fix: install() must target LMCacheMPWorkerAdapter. The r90 bug
    wrapped LMCacheMPSchedulerAdapter which has NO get_finished (mock of a
    non-existent method) -> silent SKIP, no protection at all.
    MUTATION caught: pointing install() back at SchedulerAdapter."""
    mod, worker_cls = guard_mod
    res = mod.install()
    assert res["status"] in ("ACTIVE", "IDEMPOTENT"), res
    assert getattr(worker_cls, "_kvp_race_guard", False), (
        "install() must wrap LMCacheMPWorkerAdapter (the real get_finished owner)")
    fx = json.loads(FIXTURE.read_text())
    assert "get_finished" in fx["LMCacheMPWorkerAdapter"]
    assert "get_failed_request_ids" in fx["LMCacheMPWorkerAdapter"]
    assert "get_finished" not in fx.get("LMCacheMPSchedulerAdapter", {}), (
        "fixture drift: SchedulerAdapter gained get_finished -- review the patch target")


def test_late_send_across_polls_is_still_filtered(guard_mod):
    """THE key r90b scenario: req retrieve fails -> upstream DRAINS
    _failed_request_ids via get_failed_request_ids(); a LATER get_finished
    returns req in finished_sending -- the guard's PERSISTENT set must still
    drop it. MUTATION caught: dropping the persistent set / intercept -> FAIL."""
    mod, worker_cls = guard_mod
    mod.install()
    a = worker_cls()
    a._pending_sends = {"req", "other"}
    a._pending_recvs = set()
    a._failed_request_ids.add("req")
    outs = a.get_failed_request_ids()
    assert outs == {"req"}
    assert a._failed_request_ids == set(), "upstream really wiped its set here"
    got_sends, _ = a.get_finished({"req", "other"})
    assert "req" not in got_sends, (
        "late finished_sending for a failed-retrieve id must be dropped even "
        "AFTER the upstream _failed_request_ids set was drained")
    assert "other" in got_sends, "clean ids pass through untouched"


def test_same_call_sequence_also_filtered(guard_mod):
    """The reviewed race sketch: retrieve -> False AND engine finished land in
    the SAME get_finished call -- upstream registers the failure inside the
    call (emit _fail_this_call) while the same call reports a late
    finished_sending for it; the guard must drop that id.
    MUTATION caught: removing the before/after registration of new failures
    (the delta snapshot) or the late-send filter."""
    mod, worker_cls = guard_mod
    mod.install()
    a = worker_cls()
    a._pending_sends = {"r1", "keep"}
    a._fail_this_call = {"r1"}           # the retrieve fails within THIS call
    got_sends, _ = a.get_finished({"r1", "keep"})
    assert "r1" not in got_sends, (
        "a retrieve failing in the same get_finished call must still be "
        "filtered out of its finished_sending return")
    assert "keep" in got_sends


def test_normal_requests_unaffected(guard_mod):
    mod, worker_cls = guard_mod
    mod.install()
    a = worker_cls()
    a._pending_sends = {"s1", "s2"}
    got_sends, _ = a.get_finished({"s1", "s2"})
    assert got_sends == {"s1", "s2"}


def test_install_idempotent(guard_mod):
    mod, _ = guard_mod
    assert mod.install()["status"] == "ACTIVE"
    assert mod.install()["status"] == "IDEMPOTENT"


def test_fail_soft_without_adapter(guard_mod, monkeypatch):
    """No LMCache API -> SKIP, never raises (engine must still launch)."""
    monkeypatch.setattr(
        sys.modules["lmcache.integration.vllm.vllm_multi_process_adapter"],
        "LMCacheMPWorkerAdapter",
        object,
    )
    mod, _ = guard_mod
    res = mod.install()
    assert res["status"] == "SKIP", res
