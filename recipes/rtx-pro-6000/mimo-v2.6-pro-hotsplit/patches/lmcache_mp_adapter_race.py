"""lmcache_mp_adapter_race.py -- round 90 C2: guard for the upstream race in
LMCache's vLLM multi-process adapter (LMCache #4185, vLLM #49089, fix pending
in LMCache PR #4136 -- NOT merged at verification time).

Race: a retrieve returning False makes vLLM finish and delete the request via
`finished_recving` (get_finished's second return). The same request_id may
then (same call or later) also appear in `finished_sending` (get_finished's
first return, `ret_stores`) -- the vLLM scheduler asserts
`req_id in self.requests` in `_update_from_kv_xfer_finished` and the engine
dies (EngineDeadError, every request 500).

Installed by sitecustomize.py AFTER lmcache.integration.vllm is imported
(import-time patch, version-guarded):
  * get_finished (healthy branch): ids in `_failed_request_ids` (a retrieve
    for them FAILED earlier -- vLLM already freed them via finished_recving)
    are removed from ret_stores before returning;
  * get_finished_with_lazy_offload: same filter only affects the second
    return (finished_retrieves) upstream -- here the first return is None,
    so just the failed-set bookkeeping is kept consistent.

The patch is safe to load on ANY LMCache build: the wrap only fires when the
adapter class exposes `get_finished` AND `_failed_request_ids` (the guard
attrs), otherwise `install()` logs a warning and does nothing (fail-soft --
the launch continues; HOOK STATUS line records the outcome).
"""
from __future__ import annotations

import functools


def _filter_failed_sends(self, ret_stores, finished_retrieves):
    """Return ret_stores MINUS any request_id whose retrieve already failed
    (`_failed_request_ids`). Those were reported to vLLM as finished_recving
    -- vLLM deleted the request; a later finished_sending for the same id is
    exactly the race that kills the scheduler."""
    failed = getattr(self, "_failed_request_ids", None)
    if not failed or not ret_stores:
        return ret_stores
    cleaned = ret_stores - failed
    if cleaned != ret_stores:
        dropped = sorted(ret_stores - cleaned)
        # one line per occurrence: never silent, never loud
        print(f"lmcache_mp_adapter_race: dropped finished_sending for "
              f"failed-retrieve request(s): {dropped}", flush=True)
    return cleaned


def install() -> dict:
    """Wrap get_finished on LMCacheMPSchedulerAdapter. Returns a status dict;
    never raises. Idempotent (a second install is a no-op)."""
    try:
        from lmcache.integration.vllm import vllm_multi_process_adapter as mod
    except Exception as e:
        print(f"lmcache_mp_adapter_race: SKIP -- lmcache adapter not importable ({e!r})")
        return {"status": "SKIP", "reason": repr(e)}

    cls = getattr(mod, "LMCacheMPSchedulerAdapter", None)
    if cls is None or not hasattr(cls, "get_finished"):
        print("lmcache_mp_adapter_race: SKIP -- adapter API not recognised (LMCache build differs)")
        return {"status": "SKIP", "reason": "no LMCacheMPSchedulerAdapter.get_finished"}
    if getattr(cls, "_kvp_race_guard", False):
        return {"status": "IDEMPOTENT"}

    _orig_get_finished = cls.get_finished

    @functools.wraps(_orig_get_finished)
    def _get_finished(self, finished_req_ids_from_engine):
        ret_stores, finished_retrieves = _orig_get_finished(self, finished_req_ids_from_engine)
        if ret_stores is None:
            return ret_stores, finished_retrieves
        return _filter_failed_sends(self, ret_stores, finished_retrieves), finished_retrieves

    cls.get_finished = _get_finished
    cls._kvp_race_guard = True
    print("lmcache_mp_adapter_race: ACTIVE (get_finished wrapped -- finished_sending "
          "is filtered against _failed_request_ids; upstream fix #4136 not merged)")
    return {"status": "ACTIVE"}

