"""Persistent KV cache integration (kv-persistence) — contract tests.

The recipe launchers carry an OPT-IN integration with the external LMCache
cache service (github.com/sikkatu/kv-persistence). The contract:

* OFF by default: with KV_PERSIST unset or 0 the launchers behave exactly as
  before — no connector, no new arguments, nothing changes.
* ON (KV_PERSIST=1): the launchers append ONLY `--kv-transfer-config` pointing
  at the cache service; image, patches and every other argument stay identical.
* The knob block is present in all three tp4.env.example locales with the same
  variable names, values and order (locale-parity contract).
* No secrets: only host/port knobs exist; no key material in the block.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit"
LAUNCHERS = [BASE / "scripts" / "run-tp4-uva.sh", BASE / "scripts" / "run-tp4-hotsplit.sh"]
LOCALES = [BASE / "tp4.env.example", BASE / "tp4.env.example.ru", BASE / "tp4.env.example.zh"]


def test_block_present_in_all_locales():
    for p in LOCALES:
        text = p.read_text()
        for knob in ("KV_PERSIST", "KV_PERSIST_HOST", "KV_PERSIST_PORT"):
            assert re.search(rf"# {knob}=", text), f"{p.name}: knob {knob} missing"


def test_locale_parity_of_values():
    """Same names, same values, same order in all three languages."""
    assign = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")

    def kv_persist_pairs(text: str):
        out = []
        for line in text.splitlines():
            m = assign.match(line)
            if m and m.group(1).startswith("KV_PERSIST"):
                out.append((m.group(1), m.group(2).split("#")[0].strip()))
        return out

    ref = kv_persist_pairs(LOCALES[0].read_text())
    assert ref, "no KV_PERSIST knobs in the English reference"
    for p in LOCALES[1:]:
        got = kv_persist_pairs(p.read_text())
        names_got = [n for n, _ in got]
        names_ref = [n for n, _ in ref]
        assert names_got == names_ref, f"{p.name}: knob order differs: {names_got} vs {names_ref}"
        assert got == ref, f"{p.name}: values differ: {got} vs {ref}"


def test_off_by_default_in_launchers():
    for p in LAUNCHERS:
        text = p.read_text()
        assert 'KV_PERSIST="${KV_PERSIST:-0}"' in text, f"{p.name}: KV_PERSIST must default to 0"
        assert text.index('KV_PERSIST="${KV_PERSIST:-0}"') < text.index('docker run -d --name'), (
            f"{p.name}: knob must be resolved before docker run")


def test_blocks_syntax_valid():
    for p in LAUNCHERS:
        subprocess.run(["bash", "-n", str(p)], check=True)


def test_connector_config_shaped_correctly():
    for p in LAUNCHERS:
        text = p.read_text()
        assert "LMCacheMPConnector" in text, f"{p.name}: connector missing"
        assert "kv_connector_module_path" in text, f"{p.name}: module path pin missing"
        assert "kv-transfer-config" in text and '"$KV_TRANSFER_CONFIG"' in text, (
            f"{p.name}: --kv-transfer-config must be passed as ONE array element")
        # host/port must be validated before being pasted into JSON
        assert "unexpected characters" in text, f"{p.name}: no sanitisation of host/port"


def test_no_secrets_in_block():
    pat = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*['\"]?[A-Za-z0-9]{8}")
    for p in LOCALES + LAUNCHERS:
        for line in p.read_text().splitlines():
            if "KV_PERSIST" in line:
                assert not pat.search(line), f"{p.name}: possible credential near KV_PERSIST: {line}"


def test_off_state_adds_no_arguments():
    """Behaviour: the block only APPENDS arguments inside the KV_PERSIST=1 branch."""
    for p in LAUNCHERS:
        text = p.read_text()
        seg = text[text.index('KV_PERSIST="${KV_PERSIST:-0}"'):text.index('docker run -d --name')]
        # every MODEL_ARGS mutation must be inside the if-branch, not at top level
        outside = [ln for ln in seg.splitlines()
                   if "MODEL_ARGS+=" in ln and not ln.startswith("  ")]
        assert not outside, f"{p.name}: MODEL_ARGS modified outside KV_PERSIST branch: {outside}"