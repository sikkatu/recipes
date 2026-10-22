# 4x RTX PRO 6000 Recipes

> **English** | [Русский](README.ru.md) | [中文](README.zh.md)


Hardware-pinned, verified recipes for serving large open-weight LLMs on **4 x RTX PRO 6000
(96 GB GDDR7, TP4, PCIe 4.0 host slots (cards are Gen 5), no NVLink)**.

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
| `rtx-pro-6000` | [MiMo-V2.6-Pro — 4× RTX PRO 6000 (TP4): UVA expert offload + hotsplit](recipes/rtx-pro-6000/mimo-v2.6-pro-hotsplit/) | MiMo-V2.6-Pro | ✅ verified | — | — |

<!-- recipe-index:end -->

## What this repo is about

The four recipes below balance **routed-expert residency** on the workstation: which MoE experts
sit in GPU memory and which are read from pinned host memory over PCIe. The balance is computed with one
shared, unit-tested planner (greedy hot set by routing share per byte) sized per model geometry.

**Full guide lives in the [Wiki](https://github.com/sikkatu/recipes/wiki):** [Quick Start](https://github.com/sikkatu/recipes/wiki/Quick-Start) · [How Expert Balance Works](https://github.com/sikkatu/recipes/wiki/How-Expert-Balance-Works) · [Models & Geometry](https://github.com/sikkatu/recipes/wiki/Models-and-Geometry) · [FAQ](https://github.com/sikkatu/recipes/wiki/FAQ)

## How a recipe is laid out

```
recipes/<hardware>/<recipe>/
  README.md          current state: what it runs, hardware, software, launch, verify, results, known limits, rollback
  balance.json       expert geometry (layers, experts, top-k, per-expert bytes, TP)  [balance recipes]
  tp4.env.example    every knob, with the default
  scripts/           launch / preflight / balance / check — what was actually run
  patches/           code mounted into the container; the balance calculator and planner
```


## Contract

1. No `:latest` — image digests, always.
2. Model revisions are commit SHAs.
3. Hardware is what `nvidia-smi` says — not the spec sheet.
4. Status labels are honest: `verified` means it ran end-to-end on the recorded host; a recipe that has not run says so.
5. Known limits are listed, or you didn't look.
6. Rollback is a script.
7. No credentials in the tree.

## Attribution

These recipes are the product of substantial hands-on work on real hardware. If you use this repository — its recipes, measurements, write-ups or documentation — in your own work, **reference it**: link `https://github.com/sikkatu/recipes` (and the specific recipe page) wherever you publish results derived from it. Forks and derived documents must keep this reference. Issues and corrections are welcome; silent copying is not.

## Verify locally

```bash
uv venv .venv && uv pip install -p .venv/bin/python jsonschema pyyaml pytest numpy
.venv/bin/python -m pytest tests -q && .venv/bin/python scripts/check_recipe.py --all
```
