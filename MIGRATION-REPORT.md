# Migration report — finetune + training crates from brainwires-framework

Status: rullama main, commit `55b077f`. Sister commit: brainwires-framework
`v-0.11` at `f1ae58bd`.

This file is a working reference. Not committed. Safe to delete once Option 3
is underway.

---

## What landed

Two new crates in the rullama workspace:

- `crates/rullama-finetune/` — vendored from brainwires-framework's
  `brainwires-finetune-local`. Burn 0.20 + safetensors + tokenizers. ~5,200
  LOC. Native-only (`#![cfg(not(target_arch = "wasm32"))]`). A `shared/`
  submodule vendors `config` / `error` / `types` from `brainwires-finetune`
  so the crate has zero `brainwires-*` deps.
- `crates/rullama-training/` — vendored from `brainwires-training`. ~15-line
  placeholder docstring.

Workspace conversion in the same commit: `src/` and `examples/*.rs` moved
to `crates/rullama/`, package name kept as `rullama` so PWA + iOS bench
import sites and wasm-pack output filenames (`pkg/rullama.{js,bg.wasm}`)
were unchanged.

Migration mechanics verified clean: cargo check on native + wasm32, 47
fast unit tests pass, wasm-pack bundle reproduces, Dockerfile path math
correct, ios-bench path dep resolves, no `unsafe`, no stale references to
the pruned `TrainingError::Dataset` / `TrainingError::Http` variants.

---

## What the code review found

Decision was to call out the imported code's quality before relying on it.
The vendoring is fine; the vendored content is not.

### Show-stoppers — the named methods don't do what their names say

1. **Training loss is MSE on token IDs cast to float**, not cross-entropy
   on logits.
   - `src/burn_backend/batch.rs:25-29` — `target = (tok as f32 / 128.0) - 1.0`
   - `src/burn_backend/training.rs:94-95` — `(output - target).powf_scalar(2.0).mean()`
   - The `cross_entropy_loss` function exists at `src/burn_modules.rs:234`
     and is never called.
   - Every "trained" adapter is meaningless.

2. **DPO / ORPO "log-probs" are negative MSE.**
   - `src/burn_backend/alignment.rs:100-110, 263-284` — comment literally
     says "proxy for actual log-probs".
   - β·(MSE_chosen − MSE_rejected) is not a policy log-ratio. Whatever
     gradient this produces, it isn't DPO or ORPO.

3. **Trainer synthesizes one `LoraLinear(dim, dim)` and ignores
   `config.lora.target_modules`.**
   - `src/burn_backend/training.rs:30-47, 192-211`
   - `src/burn_backend/weights.rs:118` filter (`shape[0] == dim && shape[1] == dim`)
     silently falls through to random init for non-square projections (GQA
     k/v) and for `.gguf` files (only `.safetensors` accepted at
     `weights.rs:100`).

4. **Adapter saved as opaque `[u64-le-len][bytes][u64-le-len][bytes]…` blob.**
   - `src/burn_backend/weights.rs:42, 55` writes `adapter_weights.bin`
     with no version / shape / dtype / tensor names.
   - `CheckpointManager::load_weights` only reads SafeTensors. Nothing in
     the crate can read this format back.

5. **`Box::leak` on every checkpoint save and every SafeTensors export.**
   - `src/checkpointing.rs:128-130`, `src/export.rs:112, 145`.
   - Permanent allocation per save. Long runs leak `O(params × saves)`.

6. **"INT4" quantization stores one byte per element.**
   - `src/quantization.rs:39-65` — each `u8` holds a value in 0..=15.
   - `src/adapters/qlora.rs:55-57` VRAM math (`bits*total/8`) is therefore
     a lie; reality is `total` bytes. QLoRA's memory premise is broken.

### Silent feature-flag drops

Declared in `TrainingHyperparams` (`src/shared/config.rs`), never read in
the burn_backend training paths:

- `max_grad_norm` — no gradient clipping happens.
- `gradient_accumulation_steps` — every micro-batch is stepped immediately.
- `gradient_checkpointing` — flag has no effect.
- `mixed_precision` — runs are f32 regardless.
- `seed` — no RNG plumbing, runs are non-deterministic.

Other dropped configs:

- `LoraConfig.dropout` — not applied in `LoraLinear` / `QLoraLinear`
  forward (`src/burn_modules.rs:120-132, 526-535`).
- `BurnBackend::train` ignores `config.device` and always uses
  `WgpuDevice::default()`. `available_devices()` returns a hardcoded list
  rather than probing wgpu adapters.

### Smaller issues

- `dataset_loader.rs:232` — targets are not `[1..]`-shifted for next-token
  prediction. If the loss were fixed, targets would still be off-by-one.
- `dataset_loader.rs:155-158` — chat-format parsing overwrites instead of
  appending; multi-turn assistant traces silently lose earlier turns.
- `dataset_loader.rs:50` — BOM not stripped before `serde_json::from_str`.
- `dataset_loader.rs:213-219` — `SimpleTokenizer::vocab_size = 257` but
  only 0..=255 are ever emitted; the spare 2 tokens are dead.
- `weight_loader.rs:168-189` — `load_config` reads HF-style keys from
  safetensors metadata, which is almost always empty in practice. Falls
  back to `dim = rank * 64` (`training.rs:38-40`), producing shape
  mismatches that quietly disable the SafeTensors weight path.
- `lr_schedule.rs:53-54` — cosine decay isn't clamped at `progress = 1.0`;
  an off-by-one at end-of-training swings LR back up.
- `burn_modules.rs:683` — `BurnTransformerBlock` runs "attention" on a 2D
  `[batch, hidden]` tensor with no sequence dim, so `softmax(q·kᵀ/√d)`
  mixes unrelated batch examples together. Demo-only struct; unused in the
  training paths today, but flagged for awareness.
- `tests/local_training_integration.rs` is gated on `#[cfg(feature = "local")]`;
  the Cargo manifest doesn't define `local`. ~500 lines of integration
  tests are effectively `#[cfg(false)]`.
- Two wgpu versions in `Cargo.lock`: 29.0.3 (rullama-core) and 26.0.1
  (pulled by burn-wgpu 0.20). They can't share `wgpu::Instance` or adapter
  handles. Relevant when wiring the future fine-tune → inference handoff.

### What was clean

- No `unsafe`. No stale brainwires deps.
- The wasm32 cfg gate works — crate body is empty for wasm32.
- f16 / bf16 conversion helpers in `weight_loader.rs:234-267` are correct.
- `LoraLinear::init` zero-inits B so the initial LoRA contribution is zero.
- Vendoring mechanics: clean. The two pruned `TrainingError` variants had
  no remaining callers; the import rewrites are complete and consistent.

---

## Decision: Option 3 — rewrite training on rullama's native wgpu kernels

Original three options:

1. Keep the vapor as-is, file bugs, mark preview. _(rejected)_
2. Tear it down to an honest subset (keep `shared`, `lr_schedule`,
   `dataset_loader`, `weight_loader`; delete the rest). _(rejected)_
3. Build training natively on rullama's existing wgpu kernels. _(chosen)_

This was already listed in the migration plan's "Out of scope (future
milestones)" section as the long-term direction. The review just moved
its priority up.

### Concrete state going into Option 3

What rullama already has and can be reused for training (in `crates/rullama/src/`):

- Kernels: matmul (Q4_K / Q6_K / F16 / F32), rmsnorm, rope, geglu,
  softcap, attention. All have CPU oracle implementations under
  `reference/` and are already parity-tested.
- Forward pass: `reference/forward_chained.rs` — full Gemma 4 forward
  on GPU.
- Weights / KV cache: `backend/weight_cache.rs`, GPU-resident state.
- Sampling: `sampling.rs`.
- Tokenizer: `tokenizer/bpe.rs` — bit-exact vs Ollama.
- GGUF v3 parser: `gguf/`.

What's missing for training:

- Reverse-mode autodiff over the existing wgpu kernels. There is no
  gradient tape, no backward kernels.
- An optimizer. SGD / Adam state lives in GPU buffers that don't exist
  yet.
- A LoRA adapter that hooks into specific projection matmuls (q_proj,
  k_proj, v_proj, o_proj) without forking the full forward.
- A proper next-token-prediction loss (cross-entropy on the logits
  rullama already produces).
- Checkpoint save / load in a real format (probably safetensors with
  full tensor metadata, not the homegrown blob).

### Suggested ordering for Option 3

Not a plan, just shape:

1. **Pick a minimum target.** SGD on a single LoRA adapter on `attn_q` of
   layer 0. Single-batch, single-token, no scheduler, no checkpointing.
   Cross-entropy loss on the next token.
2. **Write the backward kernels you need for that minimum target.** That
   forces the autodiff seam to be designed against a real example
   instead of in the abstract.
3. **Generalize to all attention projections, then to FFN.**
4. **Add optimizer state (Adam), gradient clipping, grad accumulation,
   mixed precision — actually wired, not declared and ignored.**
5. **Checkpoint format: real safetensors, named tensors, dtype, shapes.**
6. **Then rebuild the public API.** Cargo deps shrink: drop `burn-core`,
   `burn-nn`, `burn-optim`, `burn-autodiff`, `burn-wgpu`, `burn-ndarray`.
   Keep `safetensors`, `tokenizers`, `serde`, `tokio`. Keep the `shared/`
   types as the public config / error / progress surface so the API
   shape doesn't churn for downstream callers.

### What to delete from the current rullama-finetune

When Option 3 starts, the following can go without replacement:

- `src/adapters/` — LoRA / DoRA / QLoRA structs that wrap Burn types.
- `src/alignment/` — DPO / ORPO files; the math is wrong.
- `src/architectures/` — generic transformer scaffolding, unused except
  by the broken `BurnTransformerBlock`.
- `src/burn_backend/` — all of it.
- `src/burn_modules.rs` — Burn-derived LoRA / DoRA / QLoRA modules,
  losses, the broken transformer block.
- `src/quantization.rs` — the 1-byte-per-int4 implementation.
- `src/checkpointing.rs` — `Box::leak` checkpoint manager.
- `src/export.rs` — `Box::leak` exporter.
- `src/weight_loader.rs` — only kept if Option 3 still wants safetensors
  ingestion; replace before relying on it.
- `tests/local_training_integration.rs` — entire dead `#[cfg(feature = "local")]`
  block.

What's worth keeping during the rewrite:

- `src/shared/` — config / error / types vendored from brainwires-finetune.
  Honest scaffolding; the structs themselves are fine.
- `src/lr_schedule.rs` — clamp the cosine progress at 1.0 and it's done.
- `src/dataset_loader.rs` — fix the next-token shift (line 232), the
  multi-turn overwrite (line 155-158), the BOM (line 50), and the dead
  spare tokens (line 213-219). Otherwise structurally fine.

### When Option 3 ships

Workspace dep table shrinks substantially:

```
Before (workspace.dependencies):       After:
  burn-core                              —
  burn-nn                                —
  burn-optim                             —
  burn-autodiff                          —
  burn-wgpu                              —
  burn-ndarray                           —
  tokenizers                             tokenizers
  safetensors                            safetensors
  anyhow                                 anyhow
  tracing                                tracing
  uuid                                   uuid
  chrono                                 chrono
  tokio                                  tokio
  tempfile (dev)                         tempfile (dev)
```

Two-wgpu-version problem also goes away — only `wgpu = "29.0.3"` remains.

---

_End of report._
