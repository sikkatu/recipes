# Runtime enforcement seam

This recipe's *balance* (which experts go to GPU memory vs pinned host) is model-independent and fully
checked: `hotsplit_plan.py` + `hotsplit_balance.py` + `balance.json`. What is model-specific is the
SGLang (NVFP4 routed experts) hook that **enforces** the split at weight-load time.

| model | start from | what it does |
|---|---|---|
| GLM-5.3-Flash | `—` | SGLang serves this model with a different MoE pipeline (grouped GEMM), so there is no vLLM-style apply() to monkeypatch. The balance here is the input to a future SGLang-side implementation (a grouped-GEMM bank split), not a drop-in patch. |

There is no drop-in hook to start from for this engine; see the balance README for what an
implementation would have to do.
Rebase the seam onto your exact image, syntax-check, and run with the plan-only flag
(`HOTSPLIT_DRYRUN=1`) first.

**Nothing in this directory is a validated runtime hook.** Until the port is done and logged, the
supported path is the plain UVA baseline (`../scripts/run-tp4-uva.sh`).
