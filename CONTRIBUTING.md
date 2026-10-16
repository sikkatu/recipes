# Contributing a recipe

> **English** | [Русский](CONTRIBUTING.ru.md) | [中文](CONTRIBUTING.zh.md)


A recipe is a promise: *on this exact hardware, with this exact software, this command produces these numbers, measured this way.* Everything in this file exists to keep that promise checkable by a stranger with the same box.

## The contract

Every recipe lives at `recipes/<hardware-slug>/<recipe-slug>/` and contains:

| file | purpose |
|---|---|
| `recipe.yaml` | **Required for `experimental`/`verified`/`deprecated` recipes; a `draft` may omit it** (then the index shows it as draft from its README H1, and `check_recipe.py` refuses any non-draft claim in a draft README). Machine-readable pin: model revision, image digest, hardware as **observed**, launch command, metrics with provenance, gates, limits, rollback. Validated by `schemas/recipe.schema.json` + `scripts/check_recipe.py` in CI. |
| `README.md` | the human page. Current state only — no diary. Required sections: `What this runs`, `Hardware`, `Software`, `Launch`, `Verify`, `Results`, `Known limits`, `Rollback`. |
| `scripts/` | the launch script the README shows, plus health wait, warmup, benchmark, rollback. What you ran, not a tidied version of it. |
| `results/<run-id>/` | provenance bundle for every published number: `system.json` (from `scripts/collect_system_snapshot.sh`), image digest, model revision, exact launch command, the raw benchmark/quality output the number was read from, trimmed logs. | Standard-bench rows (T1 perf / T2 quality) use the pinned bench and tiers in [docs/benchmarking.md](docs/benchmarking.md).
| `known-limits.md` (optional) | longer form of `limits:` when the tradeoffs need explaining. |
| `research/` (optional) | ledgers, root-cause writeups, failed experiments. Encouraged — what didn't work is half the value. |
| `patches/` (optional) | any code the recipe mounts into the container; each listed in `recipe.yaml` with its sha256. |

## Rules the checker enforces

- **No `:latest`.** Container images are `name:tag` plus a `sha256:` digest.
- **Model revisions are 40-char commit SHAs**, never branch names.
- **Hardware is what the machine reports**, not the spec sheet. `hardware.observed` must point at a `results/<run-id>/system.json` that produced it. (The four RTX PRO 6000 cards report 96 GB each as the machine sees it, not the marketing number. Say so.)
- **Every metric has provenance**: `run_id`, `method`, and the `raw` file the number came from — and that file must exist.
- **`status: verified` requires** every gate (`schema`, `digest`, `health`, `quality`, `performance`) to carry a `last_pass_utc` and `run_id`, and at least one published metric.
- **No credentials** anywhere in the tree. API keys are read from files outside the repo at launch time.
- **Patches match their sha256.**

## Rules humans enforce (PR review)

- A number without a workload shape (concurrency, prompt/output lengths, prompt class) is not a number.
- If a number was measured before a fix, it carries a `label:` saying so and the README says so next to it.
- Quality gates are **per recipe and stated**. One recipe may accept FP8 KV cache; another may forbid it. Neither is "the" bar — the bar is whatever the recipe declares and measures.
- `Known limits` is not optional in spirit: if you didn't find any, you didn't look.
- Rollback must be a script, not a sentence.

## Status meanings

| status | means |
|---|---|
| `draft` | being written; may not launch |
| `experimental` | launches and has numbers, but a gate is unmeasured or a known defect is open |
| `verified` | all five gates passed on a recorded run; a stranger with the box should reproduce the headline within ~5% |
| `deprecated` | superseded or the software has moved on; kept for the record |

## Workflow

```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy pillow
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/check_recipe.py --all
.venv/bin/python scripts/render_index.py
```

Open a PR; CI runs the same four commands (pytest, check_recipe, render_index --check, plus the
fork-trace scan in the test suite). Reviews focus on provenance, not prose.

## Attribution

Recipes stand on other people's kernels, engines, and checkpoints. Credit the authors of what you used, by name and link, in the recipe README. Measure before arguing with them.

## Agent claim card (`card.yaml: harness:`)

`card.yaml` is **required only when the recipe makes agent/tool-use claims** that need a claim card
(BFCL, flood fixture, tool-call acceptance). For pure throughput/latency recipes it is optional;
`check_recipe.py` validates it only when present.

A tok/s number plus a hardware list is not a claim about agents. When a recipe claims agent/tool-call quality it carries, on the card:

- **Fixed outer protocol** — `harness/protocol.yaml` in the recipe dir (sampling, tool_choice, retries, concurrency, turns, system prompts, the *solved* definition, suite sizes, cost formula and $/kWh assumption), with its sha256 on the card. Any change = new file, new hash, new run.
- **Dev suite** — run during development; decides *candidacy* only.
- **Held-out sibling** — same shape, frozen at creation, never run inside a campaign, run once at promotion; decides *promotion*. A candidate that passes dev and fails held-out is recorded as overfit-to-harness and is not re-run against the sibling.
- **Cost per solved task, printed as a pair with Pass@1** — `Pass@1 X% · $Y / 1000 solved` on one line, so a cost claim cannot hide a quality drop; denominator = verifier-passed tasks on the frozen held-out split; **price date stamped** (`priced_on`) because $/kWh and $/hr drift; both bases stated (energy; amortized with its $/hr). The unsolved breakdown (no-call / truncated / error / wrong-arg) prints under it so a model that bails early is visible, not just neutralised.

Unmeasured fields render amber `pending`. Same-grader A/B only — never place these numbers beside a public leaderboard.
