# Runtime enforcement seam

> **English** | [Русский](README.ru.md) | [中文](README.zh.md)


This recipe's *balance* (which experts go to GPU memory vs pinned host) is model-independent and fully
checked: `hotsplit_plan.py` + `hotsplit_balance.py` + `balance.json`. What is model-specific is the
vLLM .28 (NVFP4 routed experts) hook that **enforces** the split at weight-load time.

| model | start from | what it does |
|---|---|---|
| GLM-5.3 | `../mimo-v2.6-pro-hotsplit/patches/hotsplit.py` | Same TP-aware hot/cold bank split. For NVFP4 the seam is the Nvfp4MoEMethod apply path; a per-layer GPU-resident expert slot cache (a fixed count of hot slots per layer) is the functionally equivalent design. |

Start from the MiMo-V2.6-Pro TP4 hook `../mimo-v2.6-pro-hotsplit/patches/hotsplit.py` — it shows the TP4 shape of the same
design (per-rank byte budget, static + dynamic hot<->cold swaps, NUMA pinning).
Rebase the seam onto your exact image, syntax-check, and run with the plan-only flag
(`HOTSPLIT_DRYRUN=1`) first.

**Nothing in this directory is a validated runtime hook.** Until the port is done and logged, the
supported path is the plain UVA baseline (`../scripts/run-tp4-uva.sh`).
