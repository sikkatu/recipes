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