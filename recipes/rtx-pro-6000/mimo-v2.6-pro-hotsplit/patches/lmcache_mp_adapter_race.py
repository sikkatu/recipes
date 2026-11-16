"""lmcache_mp_adapter_race.py -- r90 C2 (fixed 90b): upstream race guard.

LMCache #4185 / vLLM #49089 / LMCache PR #4136 (NOT merged at verify time).
Race: retrieve->False -> vLLM deletes the request via finished_recving; a
LATER get_finished returns the same id in finished_sending (ret_stores) ->
scheduler assert -> EngineDeadError.

r90b review fix: get_finished lives in LMCacheMPWorkerAdapter (verified by
AST scan of the real v0.5.6 source), NOT LMCacheMPSchedulerAdapter. Also the
upstream _failed_request_ids is DRAINED by get_failed_request_ids() after
every poll, so the guard keeps its OWN persistent set and intercepts the
drain to copy the ids before they are lost.

FAIL-SOFT: install() never raises; without the exact LMCache v0.5.6 API it
returns SKIP (sitecustomize prints the HOOK STATUS line).
TEMPORARY: remove once LMCache PR #4136 lands.
"""
from __future__ import annotations

import functools

_MAX_GUARD_IDS = 4096


def _remember(self, rid):
    s = getattr(self, "_kvp_failed_retrieve_ids", None)
    if s is None:
        s = set()
        self._kvp_failed_retrieve_ids = s
    s.add(rid)
    if len(s) > _MAX_GUARD_IDS:
        print("lmcache_mp_adapter_race: guard exceeded %d ids -- clearing (worst case one late send slips through)" % _MAX_GUARD_IDS, flush=True)
        s.clear()
        s.add(rid)


def _filter_late_sends(self, ret_stores):
    if not ret_stores:
        return ret_stores
    failed = getattr(self, "_kvp_failed_retrieve_ids", None)
    if not failed:
        return ret_stores
    leaked = ret_stores & failed
    if not leaked:
        return ret_stores
    for rid in leaked:
        failed.discard(rid)   # one-shot: this late send IS the guarded event
    print("lmcache_mp_adapter_race: dropped finished_sending for failed-retrieve request(s): %s" % sorted(leaked), flush=True)
    return ret_stores - leaked


def install() -> dict:
    """Wrap LMCacheMPWorkerAdapter.get_finished (+lazy) and
    get_failed_request_ids. Idempotent, fail-soft, never raises."""
    try:
        from lmcache.integration.vllm import vllm_multi_process_adapter as mod
    except Exception as e:
        print("lmcache_mp_adapter_race: SKIP -- lmcache adapter not importable (%r)" % (e,))
        return {"status": "SKIP", "reason": repr(e)}

    cls = getattr(mod, "LMCacheMPWorkerAdapter", None)
    if cls is None or not hasattr(cls, "get_finished") or not hasattr(cls, "get_failed_request_ids"):
        print("lmcache_mp_adapter_race: SKIP -- LMCacheMPWorkerAdapter API not recognised (targets the LMCache v0.5.6 surface)")
        return {"status": "SKIP", "reason": "LMCacheMPWorkerAdapter API not recognised"}
    if getattr(cls, "_kvp_race_guard", False):
        return {"status": "IDEMPOTENT"}

    orig_get_finished = cls.get_finished
    orig_get_failed = cls.get_failed_request_ids

    def _snap(self):
        return set(getattr(self, "_failed_request_ids", ()) or ())

    @functools.wraps(orig_get_finished)
    def wrapped_get_finished(self, finished_req_ids_from_engine):
        before = _snap(self)
        ret_stores, finished_retrieves = orig_get_finished(self, finished_req_ids_from_engine)
        for rid in _snap(self) - before:
            _remember(self, rid)
        return _filter_late_sends(self, ret_stores), finished_retrieves

    @functools.wraps(orig_get_failed)
    def wrapped_get_failed(self):
        outs = orig_get_failed(self)
        for rid in outs:
            _remember(self, rid)   # copy ids BEFORE the next poll wipes them
        return outs

    cls.get_finished = wrapped_get_finished
    cls.get_failed_request_ids = wrapped_get_failed

    if hasattr(cls, "get_finished_with_lazy_offload"):
        orig_lazy = cls.get_finished_with_lazy_offload

        @functools.wraps(orig_lazy)
        def wrapped_lazy(self):
            before = _snap(self)
            ret_stores, finished_retrieves = orig_lazy(self)
            for rid in _snap(self) - before:
                _remember(self, rid)
            return _filter_late_sends(self, ret_stores), finished_retrieves

        cls.get_finished_with_lazy_offload = wrapped_lazy

    cls._kvp_race_guard = True
    print("lmcache_mp_adapter_race: ACTIVE on LMCacheMPWorkerAdapter (finished_sending filtered against a persistent failed-retrieve set; upstream fix #4136 not merged)")
    return {"status": "ACTIVE"}
