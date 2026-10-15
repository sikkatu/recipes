"""Loaded in every Python process when /w is on PYTHONPATH (see scripts/run-tp4-hotsplit.sh).

Two opt-in jobs, both inert unless their env var is set:
  HOTSPLIT_NUMA=1       pin each TP worker to the NUMA node of its GPU (numa_bind.py)
  HOTSPLIT_BOOT_HOOK=1  call hotsplit.apply_hotsplit(model) after vLLM's process_weights_after_loading,
                        for images where the model file is NOT replaced by mimo_v2_hotsplit.py
                        (the replaced file carries the same hook; the launcher enables only one)
"""
import importlib.abc
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _warn(msg):
    sys.stderr.write(f"hotsplit sitecustomize: {msg}\n")


if os.environ.get("HOTSPLIT_NUMA") == "1":
    try:
        sys.path.insert(0, _HERE)
        import numa_bind

        numa_bind.install()
    except Exception as e:  # pragma: no cover
        _warn(f"NUMA binding disabled ({e!r})")


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
            return None
        orig_exec = spec.loader.exec_module
        fn = self.fn

        def exec_module(module):
            orig_exec(module)
            try:
                fn(module)
            except Exception as e:
                _warn(f"post-import hook for {fullname} failed: {e!r}")

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
    sys.stderr.write("hotsplit sitecustomize: process_weights_after_loading wrapped\n")


if os.environ.get("HOTSPLIT_BOOT_HOOK") == "1":
    sys.meta_path.insert(0, _PostImport("vllm.model_executor.model_loader.base_loader", _wrap_loader))
