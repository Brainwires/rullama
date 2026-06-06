# Phase A2 — Tier-0 per-speaker fine-tune (reconstruction-only)

> Copied verbatim from the cloning-fidelity plan for later reference. **Not starting this next.**
> Context: A1 (clean data + robust trimmed-mean aggregation) is shipped. A2 is the next rung —
> fine-tuning the user *into* the StyleTTS2 engine — only if A1 isn't enough. Phase B (VoxCPM2)
> ships regardless; A and B are the two selectable engines, not an either/or.

---

## A2 — Tier-0 per-speaker fine-tune (reconstruction-only, the strongest lever)

If A1 isn't enough, fine-tune the user *into* the engine (what truly distinguishes a preset from a
clone). Reconstruction-only — **no GAN, no FFT in the backward graph, no LSTM backward**.

- **What's optimized:** the 256-d style vector (+ optionally the decoder AdaIN bias), everything
  else frozen, minimizing **mel-L1 in decoder-mel space** on the user's clips. Reuses the
  `rullama-finetune` Adam/scratch surface.
- **Net-new backward kernels:** `conv1d_backward`, `layernorm_backward`, `adain_backward`
  (snake-backward is trivial). Each pairs with a CPU oracle + GPU-vs-CPU parity test (project rule),
  in `crates/rullama/src/kernels/wgsl/` + `backend/dispatch.rs` + an `examples/` parity test.
- **Surface:** a "Refine my voice" step in `VoiceTrainPanel.tsx` running N steps, saving the refined
  style (+ AdaIN delta) to the voice library. Desktop-only. `crates/rullama-finetune/src/{session.rs,
  wasm_bindgen_api.rs}` gain a small `VoiceRefineSession`-style surface.
- **Honest ceiling:** reconstruction-only is duller than the GAN-trained original — "recognizably,
  polished you," not studio-perfect. Combined with A1's clean data it should clearly beat today's
  zero-shot single-vector clone. Phase B (the data-efficient engine) ships regardless — A and B are
  the two selectable engines, not an either/or; A0's data sweep tells users which fits their audio.
- **Exit:** mel-L1 drops over steps; A/B — refined voice vs A1 averaged voice; audibly closer.

---

## Relevant critical files (from the plan)

- `web/src/components/VoiceTrainPanel.tsx` — the "Refine my voice" step UI.
- `web/src/lib/voice-library.ts` — store reference audio + transcript as the canonical
  voice; keep raw clips for re-aggregation / the A2 fine-tune; add a per-voice `engine` field.
- `crates/rullama/src/styletts2_clone.rs`,
  `crates/rullama/src/reference/styletts2/{model.rs,style_encoder.rs}`.
- `crates/rullama-finetune/src/{session.rs,wasm_bindgen_api.rs}` — gain a small
  `VoiceRefineSession`-style surface.
- New backward WGSL under `crates/rullama/src/kernels/wgsl/` (`conv1d_backward`,
  `layernorm_backward`, `adain_backward`) + dispatchers in `backend/dispatch.rs` + `examples/`
  parity tests.

## Verification (from the plan)

- **A2:** native parity tests for each new backward kernel (GPU-vs-CPU, tight tol) + an
  `overfit_one`-style mel-L1-drops test; A/B refined vs averaged voice.
