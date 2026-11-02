"""Persistent KV cache integration (kv-persistence) — contract tests.
The recipe launchers carry an OPT-IN integration with the external LMCache
cache service (github.com/sikkatu/kv-persistence). The contract:
* OFF by default: with KV_PERSIST unset or 0 the launchers behave exactly as
  before — no connector, no namespace remount, no new arguments, nothing
  changes. The plain run needs no MODEL_DIR and no kv-persistence install.
* ON (KV_PERSIST=1) the launcher does FOUR things (review fix):
  1. mounts MODEL_DIR at /models/<cache-identity namespace> instead of /model,
     so model_config.model (the string LMCache keys L2 objects by) is unique
     per identity — incompatible revisions/dtypes/geometries can never read
     each other's KV blocks. The namespace comes from `kvpersist.cli
     model-name` (verified kv-persistence identity) or a deterministic hash of
     the KV-compatibility fields. MODEL_DIR is required (remote fetch cannot be
     isolated per revision).
  2. appends --kv-transfer-config pointing at the cache service;
  3. appends --enable-prompt-tokens-details (cached_tokens — the only
     machine-readable reuse counter; the kv-persistence acceptance run needs it);
  4. pins --block-size so the kv-persistence identity gate can verify it.
* The knob block is present in all three tp4.env.example locales with the same
  variable names, values and order (locale-parity contract).
* No secrets: only host/port/namespace knobs exist; no key material in the block.
Note: these are CONTRACT tests (they verify the launchers carry the machinery).
They do NOT prove real NVMe recovery — that needs a GPU-host run of
kv-persistence's scripts/test-recovery.sh (see kv-persistence docs/final-report.md).
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


# --- review round: cache-identity namespace + acceptance flags ----------------


def test_namespace_mechanism_present():
    """Review fix: the plain recipe mounted MODEL_DIR at /model, so EVERY
    deployment registered one LMCache model_name -- incompatible revisions /
    dtypes / geometries silently shared one L2 namespace. The KV_PERSIST block
    must mount the model under a namespace-coded path instead."""
    for p in LAUNCHERS:
        text = p.read_text()
        assert "KV_PERSIST_NAMESPACE" in text, f"{p.name}: no namespace derivation"
        assert "/models/" in text, f"{p.name}: model must be mounted under /models/<namespace>"
        assert 'MODEL_ARGS=(--model "$MODEL_MOUNT")' in text, (
            f"{p.name}: --model must point at the namespace mount, not /model"
        )
        assert "kvpersist.cli model-name" in text, (
            f"{p.name}: must prefer the verified kv-persistence identity hash"
        )
        # deterministic fallback hash must cover the KV-compatibility fields
        assert "config.json" in text and "kv_dtype" in text, (
            f"{p.name}: fallback namespace must hash config + kv dtype"
        )
        # remote fetch cannot be namespace-isolated -> must refuse, not share
        assert "requires MODEL_DIR" in text, (
            f"{p.name}: remote fetch (MODEL_DIR unset) must be refused for KV_PERSIST=1"
        )


def test_acceptance_flags_present():
    """Review fix: without --enable-prompt-tokens-details the acceptance run
    (kv-persistence test-recovery.sh) can never PASS -- cached_tokens is the
    only machine-readable reuse counter. And --block-size must be pinned so
    the identity gate has a verifiable engine argument."""
    for p in LAUNCHERS:
        text = p.read_text()
        assert "MODEL_ARGS+=(--enable-prompt-tokens-details)" in text, (
            f"{p.name}: --enable-prompt-tokens-details missing in KV_PERSIST branch"
        )
        assert "MODEL_ARGS+=(--block-size" in text, (
            f"{p.name}: --block-size must be pinned in KV_PERSIST branch"
        )


def test_off_state_needs_no_model_dir():
    """The namespace mechanism must live INSIDE the KV_PERSIST branch: the
    plain (KV_PERSIST=0) run must keep working with or without MODEL_DIR."""
    for p in LAUNCHERS:
        text = p.read_text()
        seg = text[text.index('KV_PERSIST="${KV_PERSIST:-0}"'):text.index('docker run -d --name')]
        outside = [ln for ln in seg.splitlines()
                   if "KV_PERSIST_NAMESPACE=" in ln and not ln.startswith("  ")]
        assert not outside, (
            f"{p.name}: namespace derivation outside the KV_PERSIST branch: {outside}"
        )


def test_identity_gate_reminder_present():
    """The block must tell the operator to verify the engine identity before
    trusting cached data (kv-persistence's verify-engine-identity.sh)."""
    for p in LAUNCHERS:
        text = p.read_text()
        assert "verify-engine-identity.sh" in text, (
            f"{p.name}: KV_PERSIST block must point at the identity gate"
        )