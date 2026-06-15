# Plan: Permanent tool-calling model (merge) + fast adapters (fuse LoRA)

## Context

Two related goals, both arising from the function-call LoRA work:

1. **Make a capability permanent** — bake a validated adapter into the base
   weights so it ships as a normal model (no adapter file, no LoRA at runtime).
2. **Make adapters that stay adapters fast** — today a loaded adapter costs
   **~247 extra GPU dispatches/token** (one fused LoRA dispatch per targeted
   projection × layer). That's *dispatch* overhead, not compute (LoRA is ~1%
   extra FLOPs); on browser WebGPU it's tens of ms/token. The fused kernel
   (`lora_matmul_fused.wgsl`) already cut 494→247; the next step removes the
   rest by folding LoRA into the base matmul.

These converge: **a merged model has zero LoRA dispatches → exactly base
speed.** So Part A delivers both "permanent" and "fastest" for a shipped
capability; Part B keeps *swappable* adapters near-base for the train-your-own
story.

> **HARD GATE:** do none of this until the v4 adapter is validated (clean
> tool calls on held-out prompts). No point baking in or optimizing an
> adapter we haven't confirmed is good.

---

## Part A — Merge the adapter into a permanent GGUF

**Goal:** ship `gemma4:e2b-toolcall` (or similar) — a standard gemma4 GGUF whose
weights already include the LoRA delta, plus the tool schema as its default
prompt. Loads like any model in the PWA (no adapter step), runs at base speed,
and — bonus — runs in Ollama/llama.cpp too (it's just gemma4 with adjusted
weights).

**Merge math (offline/build-time, per targeted tensor):**
`W_merged = dequant(W_q4k) + (α/r)·B·A`, stored as **F16**.
- F16 (not re-quantized Q4_K) because rullama has **no quantizer** — only
  dequant + quant-native matmul. F16 needs no quantizer AND is *more accurate*
  (no re-quant rounding; the merged value is stored directly). GGUF is per-tensor
  typed, so F16 merged tensors coexist with the untouched Q4_K tensors (same way
  Q4_K_M already mixes Q4_K + Q6_K). Only the ~5 merged tensor *types* per layer
  grow; the rest copy through verbatim → partial size bump, not a full f16 model.

**Work items:**
1. **GGUF writer** (the main new piece) — `crates/rullama/src/gguf/writer.rs`.
   GGUF-v3 serialization: header + metadata KV passthrough + tensor info table +
   aligned tensor data. Reference: the existing Python converters
   (`scripts/convert-kokoro-gguf.py`, `convert-styletts2-gguf.py`) show the
   on-disk layout; `gguf/reader.rs` is the inverse to mirror. ~1–2 days.
2. **Merge tool** — `crates/rullama-finetune/examples/merge_adapter.rs` (native,
   offline). Loads the base GGUF + the safetensors adapter, dequants each target
   tensor via the **validated** `gguf/quant.rs` dequant (critical: must match
   rullama's actual matmul math), adds `scale·B·A`, writes F16 via the new
   writer; copies all other tensors through unchanged. Small (~0.5 day) — reuses
   dequant, `lora.rs` adapter parse, and the projection-dim mapping from
   `session.rs`.
3. **Bake the prompt convention** — the capability is weights **+** the System
   schema. Make the tool schema the model's default system prefix: a
   `template/` default + a catalog flag, so users don't pass `tool-schema.txt`.
   (`web/src/lib/toolFormat.ts::TOOL_SCHEMA_PROMPT` is already the canonical text.)
4. **Distribution** — upload to R2; add a catalog entry in `web/src/lib/api.ts`
   (kind `chat`, family gemma4); the PWA loads it with no adapter path.
5. **Parity test** — `merge_parity`: merged-model greedy output == base+adapter
   output on the held-out tool prompts (exact in F32; F16 within tiny rounding).
   Also confirm it loads + runs in Ollama.

**Reuse:** `gguf/reader.rs`, `gguf/quant.rs` (dequant), `gguf/dtype.rs`
(`dequant_into_f16` helpers), `lora.rs` (adapter parse), the Python converters
(format reference only — keep the pipeline native).

**Format note:** F32 output = bit-exact to base+adapter but ~4× those tensors;
F16 = within ~1e-3, ~2×. Use **F16** unless parity demands F32.

---

## Part B — Fuse LoRA into the base matmul (fast swappable adapters)

**Goal:** an adapter that stays an adapter runs within a few % of base, by taking
the per-token extra-dispatch count from **~247 → 0** — fold the LoRA correction
into the quant-matmul dispatch instead of a separate `lora_matmul_fused` pass.

**Approach:**
1. Extend the quant-matmul kernels (`q4_k_dequant_matmul.wgsl` + the `q6_k`,
   `q5_0`, `q8_0`, `f16` variants) to **optionally** bind A/B + rank/scale and
   add the rank-r correction in the *same* dispatch (gated by a `has_lora`
   param; the A·x→B·y math already exists in `lora_matmul_fused.wgsl` — move it
   inline). Early-out path keeps the no-adapter case byte-identical.
2. Thread LoRA slots through the matmul dispatcher
   (`backend/dispatch/matmul.rs::matmul_quant_chained`) so the forward passes
   them into the matmul instead of issuing a separate LoRA dispatch.
3. Drop the standalone `lora_matmul_fused_chained` calls from the inference
   forward (`reference/forward_chained.rs`, ~8 call sites) when fusion is active.
4. **Parity test** — fused-in-matmul LoRA == separate-dispatch LoRA, numerically
   identical, on the eval set (this touches the parity-oracle surface, so it's
   the highest-care part).
5. **Perf** — measure tok/s with adapter, before/after, on browser + native.

**Risk:** touches the core matmul kernels (rullama's parity backbone) and adds a
per-tensor bind for A/B. Each quant variant needs the path + a parity test.
Bigger and riskier than Part A. **Caveat:** even fused, "near-equivalent" is
within a few % (still reading A/B + rank-r work) — only **merge** is truly
identical-to-base.

**Reuse:** `lora_matmul_fused.wgsl` math (relocate inline), the bind-group cache.

---

## Sequencing & recommendation

1. **Gate on v4 validation** (clean held-out tool calls).
2. **Part A first** — higher value, lower risk. Delivers the permanent model
   *and* base speed for the shipped capability; the GGUF writer is reusable
   infra (future merges, exports).
3. **Part B if/when swappable-adapter speed matters** — the train-your-own UX.
   Larger kernel work; do it deliberately with full parity coverage.

Not mutually exclusive — ship the merged model (A) for the out-of-box story
*and* keep adapters fast (B) for the build-your-own story.

## Verification summary
- **A:** `merge_parity` (merged == base+adapter on held-out prompts), runs in
  rullama **and** Ollama, logged file-size delta, correct tool calls end-to-end
  in the PWA with no adapter loaded.
- **B:** numerical parity (fused == separate per quant type), measured tok/s gain
  with an adapter, and **zero regression on the no-adapter base path**.

## Effort (rough)
- Part A: ~2–3 days (GGUF writer is the bulk).
- Part B: ~several days + careful per-variant parity.
