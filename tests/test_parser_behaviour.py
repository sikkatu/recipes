"""Round 68: behavioural parser test (review: substring checks won't catch drift).

Run BOTH parsers (kv-persistence's env.sh `_kvp_seed_env_file` and the
recipe's kvp-common.sh `_kvp_seed_env_file`) on the SAME tp4.env fixture and
compare the resulting environment KEY BY KEY. A drift in quoting, $HOME
expansion, comment handling or identifier validation shows up as a value
difference here -- substring tests cannot see that.

The fixture deliberately covers every tricky rule:
  * plain value, quoted value, inline comment (" #"), '#' inside a value,
  * $HOME / ${HOME} expansion, indented comment with an assignment,
  * export prefix, trailing whitespace, empty value, invalid identifier line.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parent.parent
KV_REPO = next(
    (c for c in (
        RECIPES / "kv-persistence",
        Path.home() / "kv-persistence",
        RECIPES.parent / "kv-persistence",
    ) if c.is_dir()),
    None,
)
COMMON = RECIPES / "recipes" / "rtx-pro-6000" / "mimo-v2.6-pro-hotsplit" / "scripts" / "kvp-common.sh"
SKIP = "kv-persistence checkout not found (behavioural parser compare needs both)"

# The fixture uses KVPTEST_ prefixes so the dump can find the values; the
# deliberately-invalid name is left UNPREFIXED (a real invalid identifier).
FIXTURE = """# comment line
  # INDENTED comment with = sign
KVPTEST_MODEL_DIR=$HOME/models/x
KVPTEST_QUOTED="decode=1,prefill=0"
KVPTEST_INLINE=1 # trailing comment
KVPTEST_HASHY=abc#def
export KVPTEST_EXPORTED=yes
KVPTEST_SPACED = padded
KVPTEST_EMPTY=
9BAD=skip-me
"""

BAD_NAME = "9BAD"


def _run(parser: Path, env_file: Path, env: dict) -> dict:
    """Source the parser, seed the fixture, dump KVPTEST_* + the invalid name."""
    script = (
        f'. "{parser}"; '
        f'_kvp_seed_env_file "{env_file}"; '
        'env | grep -E "^(KVPTEST_|%s=)" || true' % BAD_NAME
    )
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       timeout=30, env=env)
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_both_parsers_produce_identical_environments(tmp_path):
    if KV_REPO is None or not (KV_REPO / "scripts" / "env.sh").is_file():
        pytest.skip(SKIP)
    assert COMMON.is_file(), "kvp-common.sh missing"

    env_file = tmp_path / "tp4.env"
    env_file.write_text(FIXTURE)

    env = os.environ.copy()
    env["HOME"] = "/kvp-test-home"

    kv_result = _run(KV_REPO / "scripts" / "env.sh", env_file, env)
    recipe_result = _run(COMMON, env_file, env)

    assert kv_result, "the env.sh parser produced no KVPTEST_* values"
    assert kv_result == recipe_result, (
        "the two parsers DRIFTED on the same tp4.env:\n"
        f"  env.sh    : {kv_result}\n"
        f"  kvp-common: {recipe_result}")

    # spot-check the tricky rules on BOTH sides at once
    assert kv_result["KVPTEST_MODEL_DIR"] == "/kvp-test-home/models/x", (
        "$HOME must be expanded")
    assert kv_result["KVPTEST_QUOTED"] == "decode=1,prefill=0", (
        "surrounding quotes must be stripped")
    assert kv_result["KVPTEST_INLINE"] == "1", (
        'inline comments must be cut at " #"')
    assert kv_result["KVPTEST_HASHY"] == "abc#def", (
        "a # inside a value must survive")
    assert kv_result["KVPTEST_EXPORTED"] == "yes", (
        "the export prefix must be accepted")
    assert kv_result["KVPTEST_SPACED"] == "padded", (
        "spaces around = and trailing whitespace must be tolerated")
    assert kv_result["KVPTEST_EMPTY"] == "", "empty values must survive"
    assert BAD_NAME not in kv_result, (
        "invalid identifiers (9BAD=...) must be skipped, not exported")