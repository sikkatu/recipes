"""Round 90 C2: LMCache adapter race guard -- behavioural test on a FAKE adapter.

Mirrors the reviewed reproduction (LMCache #4185 / vLLM #49089): a retrieve
that fails (`result=False`) registers the request in `_failed_request_ids`
and vLLM fires finished_recving (get_finished's SECOND return) and deletes
the request. A LATER get_finished call may still put the same id into
finished_sending (get_finished's FIRST return, `ret_stores`) -- the vLLM
scheduler then asserts `req_id in self.requests` and the engine dies.

The guard (patches/lmcache_mp_adapter_race.py::install) wraps get_finished
and filters `_failed_request_ids` out of `ret_stores`.

MUTATIONs caught:
  * dropping the `_filter_failed_sends` call (the guard becomes a no-op) --
    the failed-retrieve id leaks into finished_sending -> test FAILS;
  * removing the idempotence / install() entirely -- the 'normal request'
    scenario deviates or install() raises -> test FAILS.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REC = Path(__file__).resolve().parent.parent
PATCHES = REC / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "patches"
GUARD = "lmcache_mp_adapter_race"


class _FakeAdapter:
    """Mimics LMCacheMPSchedulerAdapter.get_finished happy-path behaviour:
    returns (ret_stores, finished_retrieves) taken from fixed queues."""

    def __init__(self):
        self._failed_request_ids = set()

    def get_finished(self, finished_req_ids_from_engine):
        # pretend lmcache reported all three ids as stores just finished:
        # req-failed (its retrieve FAILED earlier -- must be filtered out),
        # req-ok (clean), and req-late (also clean)
        return {"req-failed", "req-ok", "req-late"}, set()


# snapshot the pristine method BEFORE any guard install() rebinds it on the
# (module-level, shared) class -- otherwise a wrapped _FakeAdapter from a
# previous test leaks into the next one and stateful asserts (IDEMPOTENT/
# DOUBLE-WRAP) flip.
_ORIG_GET_FINISHED = _FakeAdapter.get_finished


@pytest.fixture(autouse=True)
def _reset_fake_adapter_state():
    _FakeAdapter.get_finished = _ORIG_GET_FINISHED
    if hasattr(_FakeAdapter, "_kvp_race_guard"):
        del _FakeAdapter._kvp_race_guard
    yield


@pytest.fixture()
def guard_mod(monkeypatch):
    """Import the guard and provide fakes for lmcache.integration.vllm."""
    monkeypatch.syspath_prepend(str(PATCHES))
    # fake the lmcache package tree the guard imports inside install()
    pkg_lmcache = types.ModuleType("lmcache")
    pkg_int = types.ModuleType("lmcache.integration")
    pkg_vllm = types.ModuleType("lmcache.integration.vllm")
    adapter_mod = types.ModuleType("lmcache.integration.vllm.vllm_multi_process_adapter")
    adapter_mod.LMCacheMPSchedulerAdapter = _FakeAdapter
    pkg_vllm.vllm_multi_process_adapter = adapter_mod
    pkg_int.vllm = pkg_vllm
    pkg_lmcache.integration = pkg_int
    monkeypatch.setitem(sys.modules, "lmcache", pkg_lmcache)
    monkeypatch.setitem(sys.modules, "lmcache.integration", pkg_int)
    monkeypatch.setitem(sys.modules, "lmcache.integration.vllm", pkg_vllm)
    monkeypatch.setitem(sys.modules, "lmcache.integration.vllm.vllm_multi_process_adapter", adapter_mod)

    sys.modules.pop(GUARD, None)
    mod = importlib.import_module(GUARD)
    yield mod
    sys.modules.pop(GUARD, None)


def test_failed_retrieve_id_is_filtered_from_finished_sending(guard_mod):
    """The reviewed race: retrieve->False then engine finished -> the id must
    NOT be in finished_sending (ret_stores).
    MUTATION caught: removing the _filter_failed_sends filter."""
    res = guard_mod.install()
    assert res["status"] in ("ACTIVE", "IDEMPOTENT"), res

    adapter = _FakeAdapter()
    adapter._failed_request_ids.add("req-failed")   # the failed retrieve's id

    ret_stores, finished_retrieves = adapter.get_finished(set())

    assert "req-failed" not in ret_stores, (
        "failed-retrieve request_id must NOT reach finished_sending -- "
        "the vLLM scheduler asserts req_id in self.requests for it")
    assert ret_stores == {"req-ok", "req-late"}, f"clean ids unchanged: {ret_stores}"
    assert finished_retrieves == set()


def test_normal_request_passes_unchanged(guard_mod):
    """No failed retrieves -> identical behaviour, nothing filtered."""
    guard_mod.install()
    adapter = _FakeAdapter()          # _failed_request_ids empty
    ret_stores, finished_retrieves = adapter.get_finished(set())
    assert ret_stores == {"req-failed", "req-ok", "req-late"}
    assert finished_retrieves == set()


def test_install_is_idempotent(guard_mod):
    """A second install() must be a no-op and must not wrap twice (double
    wrapping would double-log or, worse, rebind the method of an already
    wrapped class)."""
    assert guard_mod.install()["status"] == "ACTIVE"
    assert guard_mod.install()["status"] == "IDEMPOTENT"


def test_install_fails_soft_without_adapter(guard_mod, monkeypatch):
    """No LMCache (a build without the adapter class) -> install() returns
    SKIP, never raises (the engine must still launch without the guard)."""
    class _NoAdapter:
        pass

    empty_mod = types.ModuleType("lmcache.integration.vllm.vllm_multi_process_adapter")
    empty_mod.LMCacheMPSchedulerAdapter = _NoAdapter

    monkeypatch.setitem(
        sys.modules,
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        empty_mod,
    )
    monkeypatch.setattr(
        sys.modules["lmcache.integration.vllm"],
        "vllm_multi_process_adapter",
        empty_mod,
        raising=False,
    )
    res = guard_mod.install()
    assert res["status"] == "SKIP", res