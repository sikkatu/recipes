# Runtime enforcement seam

This recipe's *balance* (which routed experts go to GPU memory vs pinned host) is model-independent and fully
checked: `hotsplit_plan.py` + `hotsplit_balance.py` + `balance.json`. What is model-specific is the
vLLM MoE hook that **enforces** the hot/cold split at weight-load time.

## Upstream blocker: vLLM glm5_next on SM120 (RTX PRO 6000)

GLM-5.3-Flash's architecture `glm5_next` uses rope-free sparse MLA (`qk_rope_head_dim=0`, NoPE).
On SM120 (compute capability 12.0) there is **no working attention/KV-cache path in vanilla vLLM**:

- `vllm-project/vllm#53963` — three failure modes, all ending at
  `concat_and_cache_mla: pe_dim must be 64 for fp8_ds_mla` (the fp8_ds_mla cache kernel hardcodes
  the DeepSeek rope-64 geometry; the GLM NoPE shape violates it).
- `vllm-project/vllm#55773` — same assert + a kernel-dispatch gap (GLM's kpool widens the topk
  buffer past `index_topk`; no decode kernel dispatch for that shape on FlashInfer 0.6.18).

Candidate fixes: vllm PR #55277 (FlashInfer SM120 NoPE sparse-MLA), vllm PR #55778, and
flashinfer-ai/flashinfer#4802 (native `GLM53_NOPE` d_qk=512) — **none merged upstream at the time
of writing**. A backend-local workaround (zero-pad the cache tail and query tail to rope-64,
plus 2 dispatch entries for topk=2176) has been confirmed on-GPU in the issue thread.

**Consequence for this recipe:** the UVA baseline launcher expects a vLLM build carrying one of
those fixes (or an out-of-tree SM120 NoPE patch). With vanilla upstream vLLM the engine will fail
during init with the `pe_dim` assertion — that is a known upstream gap on this exact GPU, not a
configuration error. The launcher prints this warning before every `docker run`.

## Hot/cold enforcement

The MiMo-V2.6-Pro TP4 recipe (`../mimo-v2.6-pro-hotsplit/patches/hotsplit.py`) shows the TP4 shape
of the same design (per-rank byte budget, static + dynamic hot<->cold swaps, NUMA pinning) for
MXFP4/Marlin. For this model (NVFP4 routed experts, vLLM MoE) the enforcement seam is the
`Nvfp4MoEMethod` apply path; a per-layer HBM expert slot cache is the functionally equivalent
design. Rebase the seam onto your exact patched image, syntax-check, and run with the plan-only
flag (`HOTSPLIT_DRYRUN=1`) first.

**Nothing in this directory is a validated runtime hook.** Until the port is done and logged, the
supported path is the plain UVA baseline (`../scripts/run-tp4-uva.sh`) — and even that needs the
SM120 NoPE fix above to actually boot.
