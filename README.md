# 4x RTX PRO 6000 Recipes

Hardware-pinned, verified recipes for serving large open-weight LLMs on **4 x RTX PRO 6000
(96 GB GDDR7, TP4, PCIe 4.0 host slots (cards are Gen 5), no NVLink)**. **Every number has a method.**

A recipe here promises: on *this* hardware (as the machine reports it, not the spec sheet), with
*this* container digest and model revision, *this* command produces *these* numbers, measured
*this* way — and here is the raw output they were read from. Recipes also keep the ledgers of what
didn't work, because that's half the value.

Recipes are MIT; models and engines carry their own licenses and their authors' credit.

## Index

<!-- recipe-index:start -->

| Hardware | Recipe | Model | Status | Headline | Updated |
|---|---|---|---|---|---|
| `rtx-pro-6000` | [DeepSeek-V4.1-Flash — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload](recipes/rtx-pro-6000/deepseek-v4.1-flash-hotsplit/) | DeepSeek-V4.1-Flash | 📝 draft | — | — |
| `rtx-pro-6000` | [GLM-5.3-Flash — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload](recipes/rtx-pro-6000/glm-5.3-flash-hotsplit/) | GLM-5.3-Flash | 📝 draft | — | — |
| `rtx-pro-6000` | [GLM-5.3 — 4× RTX PRO 6000 (96 GB), TP4: expert balance for UVA expert offload](recipes/rtx-pro-6000/glm-5.3-hotsplit/) | GLM-5.3 | 📝 draft | — | — |
| `rtx-pro-6000` | [MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA expert offload + hotsplit](recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/) | MiMo-V2.6-Pro | 📝 draft | — | — |

<!-- recipe-index:end -->

## What this repo is about

The four recipes below balance **routed-expert residency** on the workstation: which MoE experts
sit in GPU memory and which are read from pinned host memory over PCIe. The balance is computed with one
shared, unit-tested planner (greedy hot set by routing share per byte) sized per model geometry.

## How a recipe is laid out

```
recipes/<hardware>/<recipe>/
  README.md          current state: what it runs, hardware, software, launch, verify, results, known limits, rollback
  RUNNING.md         short launch card (drafts)
  balance.json       expert geometry (layers, experts, top-k, per-expert bytes, TP)  [balance recipes]
  tp4.env.example    every knob, with the default
  scripts/           launch / preflight / balance / check — what was actually run
  patches/           code mounted into the container; the balance calculator and planner
  results/<run-id>/  provenance bundle for every published number (once measured)
```

`hardware/` holds one profile per machine with **observed** values;
`scripts/collect_system_snapshot.sh` produces them.

## Contract, in ten lines

1. No `:latest` — image digests, always.
2. Model revisions are commit SHAs.
3. Hardware is what `nvidia-smi` says.
4. Every metric: value, unit, workload shape, run-id, method, raw file.
5. Five gates — schema, digest, health, quality, performance — with pass dates.
6. `verified` means all five passed on a recorded run.
7. Quality gates are per recipe and stated; there is no global bar.
8. Known limits are listed, or you didn't look.
9. Rollback is a script.
10. No credentials in the tree.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contract and
[schemas/recipe.schema.json](schemas/recipe.schema.json) for the fields.

## Verify locally

```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```
