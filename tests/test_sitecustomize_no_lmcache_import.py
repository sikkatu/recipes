"""Round 90c P0-1: sitecustomize must NOT import LMCache/torch in every process.

The r90c review found that `importlib.util.find_spec("lmcache.integration.vllm.
...")` EXECUTES the parent packages to resolve the dotted name -- i.e. it runs
lmcache/__init__.py, which imports torch and picks a device, in EVERY python
process of the container (API server, workers, plain `python3 -c`) before vLLM
itself wants it.

Fix: sitecustomize registers the _PostImport finder UNCONDITIONALLY and never
calls find_spec on the dotted name. The finder is passive -- the guard installs
only when the adapter module is really imported; if lmcache never loads, its
__init__ must NOT run.

Test: run sitecustomize in a subprocess with a FAKE lmcache package whose
__init__ writes a marker file; the marker must NOT appear until the adapter
module is imported. MUTATION: bring find_spec back -> the marker appears as
soon as sitecustomize runs -> test FAILS.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
SITE = REC / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "patches"


def _make_fake_lmcache(tmp: Path, marker: Path) -> Path:
    """Fake lmcache package tree: lmcache/__init__ writes MARKER on import,
    the adapter module is a plain stub (importing it is what SHOULD install
    the guard)."""
    root = tmp / "fake"
    (root / "lmcache" / "integration" / "vllm").mkdir(parents=True)
    (root / "lmcache" / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').write('init')\n")
    (root / "lmcache" / "integration" / "__init__.py").write_text("")
    (root / "lmcache" / "integration" / "vllm" / "__init__.py").write_text("")
    (root / "lmcache" / "integration" / "vllm" / "vllm_multi_process_adapter.py").write_text(
        "class LMCacheMPWorkerAdapter:\n"
        "    def get_finished(self, finished_req_ids_from_engine):\n"
        "        return set(), set()\n"
        "    def get_failed_request_ids(self):\n"
        "        return set()\n")
    return root


def test_sitecustomize_does_not_import_lmcache_until_adapter_imported(tmp_path):
    """P0-1: merely running sitecustomize must NOT execute lmcache/__init__
    (no marker). Only importing the adapter module may trigger the chain.
    MUTATION caught: re-introducing find_spec on the dotted name (it runs
    lmcache/__init__ at startup) -> the marker appears -> FAIL."""
    marker = tmp_path / "INIT_RAN"
    fake_root = _make_fake_lmcache(tmp_path, marker)

    runner = tmp_path / "runner.py"
    runner.write_text(textwrap.dedent(f"""
        import sys, os
        sys.path.insert(0, {str(fake_root)!r})      # fake lmcache first
        sys.path.insert(0, {str(SITE)!r})          # then patches (sitecustomize + guard)
        import sitecustomize                        # registers the finder; must NOT touch lmcache
        print("AFTER_SITECUSTOMIZE marker_exists=" + str(os.path.exists({str(marker)!r})))
        # now REALLY import the adapter -> this is the documented trigger
        import lmcache.integration.vllm.vllm_multi_process_adapter
        print("AFTER_ADAPTER_IMPORT marker_exists=" + str(os.path.exists({str(marker)!r})))
    """))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SITE) + os.pathsep + str(fake_root)
    r = subprocess.run([sys.executable, str(runner)],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, f"runner crashed:\n{r.stdout}\n{r.stderr}"
    lines = r.stdout.splitlines()
    assert any("AFTER_SITECUSTOMIZE marker_exists=False" in ln for ln in lines), (
        "lmcache/__init__ must NOT run at sitecustomize time (no find_spec of the "
        f"dotted name):\n{r.stdout}\n{r.stderr}")
    assert any("AFTER_ADAPTER_IMPORT marker_exists=True" in ln for ln in lines), (
        "sanity: importing the adapter SHOULD run lmcache/__init__ (the marker "
        f"proves the fake works):\n{r.stdout}\n{r.stderr}")


def test_no_find_spec_dotted_probe_in_sitecustomize():
    """Static half of the same defect: the source must not probe the dotted
    adapter name with find_spec anywhere outside the _PostImport finder.
    MUTATION caught: re-adding the availability probe."""
    src = (SITE / "sitecustomize.py").read_text()
    # every literal `find_spec("lmcache.integration...")` call is banned
    assert 'find_spec(\n        "lmcache' not in src.replace("\r", "")
    assert 'find_spec("lmcache' not in src.replace("\r", ""), (
        "sitecustomize must not resolve the dotted lmcache name at startup "
        "(it executes lmcache/__init__ -> torch in every process)")