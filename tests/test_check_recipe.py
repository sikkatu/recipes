"""Tests for scripts/check_recipe.py — each test is one rule of the recipe contract."""
import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_recipe import check_recipe  # noqa: E402

SHA = "a" * 64
REV = "b" * 40


def base_recipe() -> dict:
    return {
        "schema": "recipe/v1",
        "id": "test-hw/test-recipe",
        "title": "Test recipe for the checker",
        "status": "experimental",
        "owners": ["sikkatu"],
        "updated": "2026-09-06",
        "model": {"name": "M", "repo": "org/M", "revision": REV, "quantization": "nvfp4"},
        "hardware": {
            "profile": "hardware/test-hw.yaml",
            "observed": {
                "gpu": "G", "driver": "1.0", "cuda": "13.0", "gpu_memory_visible_gib": 1.0,
                "os": "os", "kernel": "k", "snapshot": "results/r1/system.json",
            },
        },
        "runtime": {
            "engine": "vllm", "engine_version": "0.28.0",
            "container": {"image": "img:v1", "digest": "sha256:" + SHA},
        },
        "launch": {"script": "scripts/launch.sh", "command": "bash scripts/launch.sh"},
        "metrics": [],
        "verification": {"gates": {g: {"how": "x"} for g in ("schema", "digest", "health", "quality", "performance")}},
        "limits": ["none known"],
        "rollback": {"script": "scripts/rollback.sh", "summary": "stop and start previous"},
    }


def write_recipe(tmp_path: Path, recipe: dict, readme: str | None = None, files: dict[str, str] | None = None) -> Path:
    d = tmp_path / "recipes" / recipe["id"]
    d.mkdir(parents=True)
    (d / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    (d / "README.md").write_text(readme if readme is not None else FULL_README)
    (d / "scripts").mkdir(exist_ok=True)
    (d / "scripts" / "launch.sh").write_text("#!/bin/bash\n")
    (d / "scripts" / "rollback.sh").write_text("#!/bin/bash\n")
    (d / "results" / "r1").mkdir(parents=True)
    (d / "results" / "r1" / "system.json").write_text("{}")
    (tmp_path / "hardware").mkdir(exist_ok=True)
    (tmp_path / "hardware" / "test-hw.yaml").write_text("slug: test-hw\n")
    for rel, content in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d / "recipe.yaml"


FULL_README = "\n".join(
    f"## {h}\n\ntext\n"
    for h in ("What this runs", "Hardware", "Software", "Launch", "Verify", "Results", "Known limits", "Rollback")
)


def errors_of(path: Path) -> list[str]:
    # path = <tmp>/recipes/<hw>/<recipe>/recipe.yaml -> repo root is parents[3]
    return check_recipe(path, root=path.parents[3])


def test_valid_experimental_passes(tmp_path):
    p = write_recipe(tmp_path, base_recipe())
    assert errors_of(p) == []


def test_rejects_latest_tag(tmp_path):
    r = base_recipe()
    r["runtime"]["container"]["image"] = "img:latest"
    p = write_recipe(tmp_path, r)
    assert any("latest" in e for e in errors_of(p))


def test_rejects_missing_digest(tmp_path):
    r = base_recipe()
    del r["runtime"]["container"]["digest"]
    p = write_recipe(tmp_path, r)
    assert any("digest" in e for e in errors_of(p))


def test_rejects_branch_name_as_revision(tmp_path):
    r = base_recipe()
    r["model"]["revision"] = "main"
    p = write_recipe(tmp_path, r)
    assert any("revision" in e for e in errors_of(p))


def test_metric_requires_provenance_with_existing_raw_file(tmp_path):
    r = base_recipe()
    r["metrics"] = [{
        "name": "decode", "value": 33.8, "unit": "tok/s", "workload": "C1",
        "provenance": {"run_id": "r1", "method": "bench3", "raw": "bench.txt"},
    }]
    p = write_recipe(tmp_path, r)
    errs = errors_of(p)
    assert any("results/r1/bench.txt" in e for e in errs), errs
    p = write_recipe(tmp_path.joinpath("b"), r, files={"results/r1/bench.txt": "C1 33.8"})
    assert errors_of(p) == []


def test_metric_without_provenance_rejected(tmp_path):
    r = base_recipe()
    r["metrics"] = [{"name": "decode", "value": 33.8, "unit": "tok/s", "workload": "C1"}]
    p = write_recipe(tmp_path, r)
    assert any("provenance" in e for e in errors_of(p))


def test_verified_requires_every_gate_pass(tmp_path):
    r = base_recipe()
    r["status"] = "verified"
    p = write_recipe(tmp_path, r)
    errs = errors_of(p)
    assert any("verified" in e and "gate" in e for e in errs), errs
    for g in r["verification"]["gates"].values():
        g["last_pass_utc"] = "2026-09-06T12:00:00Z"
        g["run_id"] = "r1"
    r["metrics"] = [{
        "name": "decode", "value": 33.8, "unit": "tok/s", "workload": "C1",
        "provenance": {"run_id": "r1", "method": "bench3", "raw": "bench.txt"},
    }]
    p = write_recipe(tmp_path.joinpath("b"), r, files={"results/r1/bench.txt": "C1 33.8"})
    assert errors_of(p) == []


def test_verified_requires_at_least_one_metric_with_provenance(tmp_path):
    r = base_recipe()
    r["status"] = "verified"
    for g in r["verification"]["gates"].values():
        g["last_pass_utc"] = "2026-09-06T12:00:00Z"
        g["run_id"] = "r1"
    r["metrics"] = []
    p = write_recipe(tmp_path, r)
    assert any("metric" in e for e in errors_of(p))


def test_readme_required_sections(tmp_path):
    p = write_recipe(tmp_path, base_recipe(), readme="## What this runs\n\nonly this\n")
    errs = errors_of(p)
    assert any("README" in e and "Rollback" in e for e in errs), errs


def test_id_must_match_directory(tmp_path):
    r = base_recipe()
    p = write_recipe(tmp_path, r)
    r2 = copy.deepcopy(r)
    r2["id"] = "other-hw/other"
    p.write_text(yaml.safe_dump(r2))
    assert any("directory" in e for e in errors_of(p))


def test_referenced_scripts_and_snapshot_must_exist(tmp_path):
    r = base_recipe()
    r["launch"]["script"] = "scripts/nope.sh"
    r["hardware"]["observed"]["snapshot"] = "results/r1/missing.json"
    p = write_recipe(tmp_path, r)
    errs = errors_of(p)
    assert any("scripts/nope.sh" in e for e in errs)
    assert any("missing.json" in e for e in errs)


def test_no_credentials_in_tree(tmp_path):
    p = write_recipe(tmp_path, base_recipe(), files={"scripts/env.sh": "export VLLM_API_KEY=sk-abcdef0123456789abcdef\n"})
    assert any("credential" in e.lower() for e in errors_of(p))


def test_patch_sha_must_match_file(tmp_path):
    r = base_recipe()
    r["runtime"]["patches"] = [{"path": "patches/hook.py", "sha256": SHA, "purpose": "test"}]
    p = write_recipe(tmp_path, r, files={"patches/hook.py": "print(1)\n"})
    assert any("sha256" in e and "hook.py" in e for e in errors_of(p))


def test_container_dockerfile_sha_must_match_file(tmp_path):
    import hashlib
    body = "FROM img@sha256:" + SHA + "\n"
    r = base_recipe()
    r["runtime"]["container"]["dockerfile"] = "Dockerfile.x"
    r["runtime"]["container"]["dockerfile_sha256"] = SHA
    p = write_recipe(tmp_path, r, files={"Dockerfile.x": body})
    assert any("dockerfile sha256 mismatch" in e for e in errors_of(p))
    r["runtime"]["container"]["dockerfile_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    p = write_recipe(tmp_path.joinpath("b"), r, files={"Dockerfile.x": body})
    assert errors_of(p) == []
    p = write_recipe(tmp_path.joinpath("c"), r)
    assert any("Dockerfile.x" in e and "not found" in e for e in errors_of(p))
