"""Loaded in every Python process when /w is on PYTHONPATH (see scripts/run-tp4-hotsplit.sh).

Three opt-in jobs, each with an explicit runtime state (logged as HOOK STATUS lines):
  HOTSPLIT_NUMA=1       pin each TP worker to the NUMA node of its GPU (numa_bind.py)
  HOTSPLIT_BOOT_HOOK=1  call hotsplit.apply_hotsplit(model) after vLLM's process_weights_after_loading,
                        for images where the model file is NOT replaced by mimo_v2_hotsplit.py
                        (the replaced file carries the same hook; the launcher enables only one)
  HOTSPLIT_STRICT_HOOK=1  a failure to install any requested hook is FATAL (SystemExit), not a warning.
                        Default 0 keeps the old warn-and-continue behaviour for exploratory runs.

Hook status lines, one per hook, always printed:
  HOOK STATUS <name>=ACTIVE   - installed and its callback executed
  HOOK STATUS <name>=INACTIVE - not requested (env not set); deliberate
  HOOK STATUS <name>=FAILED   - requested but installation/callback failed (reason included)

A silently-warned hook makes a dynamic-vs-static comparison unreliable: the server answers but
the hot set never adapts. With HOTSPLIT_STRICT_HOOK=1 the process refuses to continue instead.
"""
import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STRICT = os.environ.get("HOTSPLIT_STRICT_HOOK", "0") == "1"


def _log(msg):
    sys.stderr.write(f"hotsplit sitecustomize: {msg}\n")


def _fail(name, reason):
    _log(f"HOOK STATUS {name}=FAILED ({reason})")
    if _STRICT:
        _log(f"STRICT: HOTSPLIT_STRICT_HOOK=1 and hook {name!r} failed -- aborting before any weight is modified")
        raise SystemExit(f"hotsplit: required hook {name!r} failed: {reason}")
    _log(f"hook {name!r} failed ({reason}); continuing WITHOUT it (set HOTSPLIT_STRICT_HOOK=1 to make this fatal)")


def _ok(name):
    _log(f"HOOK STATUS {name}=ACTIVE")


def _inactive(name):
    _log(f"HOOK STATUS {name}=INACTIVE (not requested)")


# ---------------- NUMA binding ----------------
if os.environ.get("HOTSPLIT_NUMA") == "1":
    try:
        sys.path.insert(0, _HERE)
        import numa_bind
        numa_bind.install()
        _ok("numa_bind")
    except Exception as e:  # pragma: no cover
        _fail("numa_bind", repr(e))
else:
    _inactive("numa_bind")


# ---------------- post-import hook (HOTSPLIT_BOOT_HOOK) ----------------
class _PostImport(importlib.abc.MetaPathFinder):
    """Run fn(module) right after `name` is first imported."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            _fail("boot_hook", f"module {fullname!r} not importable")
            return None
        orig_exec = spec.loader.exec_module
        fn = self.fn

        def exec_module(module):
            orig_exec(module)
            try:
                fn(module)
                _ok("boot_hook")
            except Exception as e:
                _fail("boot_hook", repr(e))
        spec.loader.exec_module = exec_module
        return spec


def _wrap_loader(module):
    orig = module.process_weights_after_loading

    def wrapped(model, model_config, target_device):
        orig(model, model_config, target_device)
        if getattr(model, "_hotsplit_applied", False) or not os.environ.get("HOTSPLIT_COUNTS"):
            return
        sys.path.insert(0, _HERE)
        from hotsplit import apply_hotsplit
        apply_hotsplit(model)
    module.process_weights_after_loading = wrapped
    _log("process_weights_after_loading wrapped")


if os.environ.get("HOTSPLIT_BOOT_HOOK") == "1":
    sys.meta_path.insert(0, _PostImport("vllm.model_executor.model_loader.base_loader", _wrap_loader))
else:
    _inactive("boot_hook")

# ---------------- LMCache adapter race guard (r90 C2) ----------------
# See patches/lmcache_mp_adapter_race.py: get_finished() can report a LATE
# finished_sending (ret_stores) for a request whose retrieve FAILED earlier
# -> vLLM scheduler assert -> EngineDeadError. The guard filters
# _failed_request_ids out of the finished_sending set. TEMPORARY: remove
# once LMCache PR #4136 is released (upstream issue LMCache #4185,
# vLLM #49089). Uses the _PostImport finder defined above, right after
# lmcache.integration.vllm.vllm_multi_process_adapter finishes importing.
def _install_race_guard(_module):
    sys.path.insert(0, _HERE)
    # r90c P0-3: fail-closed version gate -- the launcher hashes the
    # installed LMCache adapter and only sets KVP_RACE_GUARD=1 for the exact
    # v0.5.6 build this patch was written against. Any other build starts
    # unpatched (a wrong-API wrap could crash the engine -- worse than no
    # guard at all).
    if os.environ.get("KVP_RACE_GUARD", "0") != "1":
        _inactive("lmcache_adapter_race_guard")   # version gate declined (KVP_RACE_GUARD!=1)
        return
    try:
        import lmcache_mp_adapter_race as _guard
        res = _guard.install()
        st = res.get("status", "?")
        if st in ("ACTIVE", "IDEMPOTENT"):
            _ok("lmcache_adapter_race_guard")
        else:
            _inactive("lmcache_adapter_race_guard")
    except Exception as e:
        _fail("lmcache_adapter_race_guard", repr(e))
# r90c P0-1: register the finder UNCONDITIONALLY and without find_spec.
# importlib.util.find_spec("a.b.c") EXECUTES the parent packages (a/__init__
# runs) to resolve the dotted name -- lmcache/__init__ imports torch and picks
# a device, so the old probe dragged torch into EVERY python process of the
# container (API server, workers, plain `python3 -c`) before vLLM itself
# wanted it. The _PostImport finder is passive: it only fires when the
# adapter module is really imported; if lmcache never loads, nothing happens.
sys.meta_path.insert(0,
    _PostImport("lmcache.integration.vllm.vllm_multi_process_adapter", _install_race_guard))

