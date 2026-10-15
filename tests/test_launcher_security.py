"""Launcher security + contract tests.

Contract (README.md "Contract, in ten lines" / SECURITY.md):
  - no wide binds: the API must not be exposed on 0.0.0.0 by default
  - no disabled auth: an API key is required unless the operator explicitly opts out
  - image digests, always: IMAGE must be pinned by @sha256:...
  - model revisions are commit SHAs when the checkpoint is fetched by ID
  - rollback is a script, not prose
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000"
RECIPES = ["deepseek-v4.1-flash-hotsplit", "glm-5.3-flash-hotsplit", "glm-5.3-hotsplit",
           "mimo-v2.6-pro-hotsplit"]


def _launchers(slug):
    d = BASE / slug / "scripts"
    out = [d / "run-tp4-uva.sh"]
    if slug == "mimo-v2.6-pro-hotsplit":
        out.append(d / "run-tp4-hotsplit.sh")
    return [p for p in out if p.is_file()]


@pytest.mark.parametrize("slug", RECIPES)
def test_no_wide_bind_by_default(slug):
    for p in _launchers(slug):
        text = p.read_text()
        assert ("BIND=${BIND:-127.0.0.1}" in text) or ('BIND="${BIND:-127.0.0.1}"' in text), f"{p}: bind must default to loopback"
        # the server bind must follow BIND, not be hard-coded to 0.0.0.0
        assert '--host "$BIND"' in text or "serve with your SGLang launcher" in text
        assert '--host 0.0.0.0' not in text, f"{p}: hard-coded 0.0.0.0 bind"


@pytest.mark.parametrize("slug", RECIPES)
def test_api_key_required_by_default(slug):
    for p in _launchers(slug):
        text = p.read_text()
        if "serve with your SGLang launcher" in text:
            continue  # SGLang scaffold exits before serving
        assert "API_KEY" in text, f"{p}: API key handling missing"
        assert "ALLOW_NO_AUTH" in text, f"{p}: no explicit opt-out gate for auth"
        # a non-loopback bind without a key must be refused
        assert "refusing a non-loopback bind" in text


@pytest.mark.parametrize("slug", RECIPES)
def test_image_pinned_by_digest(slug):
    for p in _launchers(slug):
        text = p.read_text()
        if "serve with your SGLang launcher" in text:
            continue
        assert 'IMAGE must be pinned by digest' in text, f"{p}: digest contract not enforced"


@pytest.mark.parametrize("slug", RECIPES)
def test_model_revision_required_when_fetched(slug):
    for p in _launchers(slug):
        text = p.read_text()
        if "serve with your SGLang launcher" in text:
            continue
        assert "MODEL_REVISION" in text, f"{p}: model revision pinning missing"
        assert 'MODEL_REVISION is required when the model is fetched' in text


@pytest.mark.parametrize("slug", RECIPES)
def test_restart_policy_not_unless_stopped(slug):
    for p in _launchers(slug):
        text = p.read_text()
        assert "--restart unless-stopped" not in text, f"{p}: never-run containers must not autostart"


@pytest.mark.parametrize("slug", RECIPES)
def test_rollback_is_a_script(slug):
    rb = BASE / slug / "scripts" / "rollback.sh"
    assert rb.is_file(), f"{slug}: rollback.sh missing (contract: rollback is a script)"
    text = rb.read_text()
    assert "docker stop" in text and "docker rm" in text
    subprocess.run(["bash", "-n", str(rb)], check=True)


@pytest.mark.parametrize("slug", RECIPES)
def test_launchers_parse(slug):
    for p in _launchers(slug):
        subprocess.run(["bash", "-n", str(p)], check=True)


def test_api_key_actually_passed_to_server():
    """Checking API_KEY in bash is not auth; it must reach the server as VLLM_API_KEY."""
    for slug in RECIPES:
        for p in _launchers(slug):
            text = p.read_text()
            if "serve with your SGLang launcher" in text:
                continue
            assert "VLLM_API_KEY=$API_KEY" in text, f"{p}: API key must be passed to the server"


def test_both_probe_requests_authenticated():
    for slug in RECIPES:
        for p in _launchers(slug):
            text = p.read_text()
            if "serve with your SGLang launcher" in text:
                continue
            for ln in text.splitlines():
                if "curl" in ln and "/v1/" in ln and ln.strip().startswith("curl"):
                    assert "Authorization: Bearer $API_KEY" in ln, f"{p}: unauthenticated probe: {ln.strip()}"


def test_exactly_one_revision_and_no_duplicate_checks():
    """Uniqueness: exactly one --revision and one of each guard (the die() check, not a comment)."""
    for slug in RECIPES:
        for p in _launchers(slug):
            text = p.read_text()
            if "serve with your SGLang launcher" in text:
                continue
            assert text.count("--revision") == 1, f"{p}: --revision must appear exactly once"
            assert text.count("IMAGE must be pinned") == 1, f"{p}: duplicate IMAGE check"
            assert text.count("MODEL_REVISION is required") == 1, f"{p}: duplicate revision check"
            # count only the real die() guard, not the explanatory comment above it
            guards = [ln for ln in text.splitlines() if "API_KEY is required" in ln and "die" in ln]
            assert len(guards) == 1, f"{p}: expected exactly 1 auth die(), got {len(guards)}"


def test_no_port_publish_under_host_networking():
    """--network host makes docker -p a no-op; the bind must come from --host, not a fake -p."""
    for slug in RECIPES:
        for p in _launchers(slug):
            text = p.read_text()
            if "serve with your SGLang launcher" in text:
                continue
            for ln in text.splitlines():
                if ln.strip().startswith("-p "):
                    pytest.fail(f"{p}: -p under --network host is a no-op: {ln.strip()}")
            assert '--host "$BIND"' in text, f"{p}: bind must follow BIND"


def test_patch_manifest_verified_before_model_replacement():
    """PATCH_MODEL_FILES replaces vLLM model files wholesale; the files must match a pinned manifest."""
    import hashlib
    p = BASE / "mimo-v2.6-pro-hotsplit" / "scripts" / "run-tp4-hotsplit.sh"
    text = p.read_text()
    assert "MANIFEST.sha256" in text, "patch manifest verification missing"
    assert "sha256sum -c" in text, "manifest must be checked with sha256sum -c"
    manifest = BASE / "mimo-v2.6-pro-hotsplit" / "patches" / "MANIFEST.sha256"
    assert manifest.is_file(), "patches/MANIFEST.sha256 missing"
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        sha, _, name = line.partition("  ")
        f = manifest.parent / name.strip()
        assert f.is_file(), f"manifest references missing file {name}"
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h == sha.strip(), f"manifest mismatch for {name}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))