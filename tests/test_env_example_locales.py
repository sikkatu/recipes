"""tp4.env.example must stay parallel across languages.

The localized files (tp4.env.example.ru / .zh) translate the COMMENTS ONLY: every
variable name, its value and its order must match the English reference exactly.
A user switching to a localized file must get byte-identical behaviour; a knob that
exists in one language but not another is a silent divergence.
"""
from __future__ import annotations
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "recipes" / "rtx-pro-6000"

# an assignment line: optional leading '# ' (the example entries are commented), NAME=...
ASSIGN = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=")


def _names(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := ASSIGN.match(line))]


def test_locales_exist_where_en_exists():
    """Every tp4.env.example ships with .ru and .zh siblings."""
    for p in sorted(BASE.glob("*/tp4.env.example")):
        for ext in (".ru", ".zh"):
            loc = Path(str(p) + ext)
            assert loc.is_file(), f"{loc} is missing (localized knob reference)"


def test_locales_have_identical_variables_in_order():
    """Same names, same values, same order in all three languages.

    Placeholder text inside <angle brackets> is language-dependent by design
    (<your key> / <ваш ключ> / <你的密钥>), so it is normalized away; anything else
    in the value -- including commented example assignments -- must match exactly.
    """
    val = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")
    placeholder = re.compile(r"<[^<>]*>")

    def pairs(text: str):
        out = []
        for line in text.splitlines():
            if m := val.match(line):
                value = m.group(2).split("#")[0].strip()
                out.append((m.group(1), placeholder.sub("<>", value)))
        return out

    for p in sorted(BASE.glob("*/tp4.env.example")):
        ref = p.read_text()
        ref_pairs = pairs(ref)
        assert _names(ref), f"{p}: no variables found"
        for ext in (".ru", ".zh"):
            loc = Path(str(p) + ext)
            loc_pairs = pairs(loc.read_text())
            assert [n for n, _ in loc_pairs] == [n for n, _ in ref_pairs], (
                f"{loc}: variables differ from the reference.\n"
                f"  ref:  {[n for n, _ in ref_pairs]}\n  loc:  {[n for n, _ in loc_pairs]}"
            )
            assert loc_pairs == ref_pairs, (
                f"{loc}: values differ from the reference: "
                f"{[(a, b) for a, b in zip(loc_pairs, ref_pairs) if a != b]}"
            )
