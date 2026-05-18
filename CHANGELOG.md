# Changelog

All notable changes to **rullama** are tracked here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays in the 0.x series, only the modules listed in the
[stability section of `lib.rs`](crates/rullama/src/lib.rs) (`api`, `error`,
`sampling`, `lora`) are covered by semver. Everything else is `#[doc(hidden)]`
and may move in any patch release.

## [Unreleased]

## [0.3.0] — 2026-05-18

In-browser fine-tuning shipped. Same `Model` you've been running inference
on can now consume a LoRA adapter the PWA trained in the foreground tab
— no native build, no separate inference engine, no round-trip through
disk. Vision-encode progress strip stays alive through the soft-token
splice and prompt prefill so the user isn't staring at a blank screen
for 2-3 min per image. Bumped to `0.3.0` because of the new `lora`
module and the scope of the fine-tune surface; the existing public API
is preserved (no signature changes, no removals).

### Public API (semver-covered modules)

- New `lora` module — `InferenceAdapter::parse_safetensors(bytes)` decodes
  the safetensors blob `TrainingSession::saveAdapter()` writes. `Model`
  owns the active adapter; consumers don't construct it directly.
- `api::Model` — additive: `has_adapter_native()` (+ JS `hasAdapter`),
  `adapter_slot_count_native()`, `load_adapter_native(bytes)` (+ JS
  `loadAdapter`), `clear_adapter_native()` (+ JS `clearAdapter`).
- `error`, `sampling` — unchanged.

### Fine-tuning (`rullama-finetune`)

- **wasm32 port.** The crate now compiles to wasm32-unknown-unknown
  (was native-only). On wasm builds the whole crate is gated off so it
  remains empty; on wasm-bindgen builds (`crate-type = ["cdylib"]`) it
  exposes a `TrainingSession` surface.
- **`TrainingSession` wasm-bindgen surface** — JS callers can run a
  forward → loss → backward → Adam step entirely in the foreground tab
  against the same wgpu kernels the inference path uses. Save trained
  weights as a safetensors `Uint8Array` and load them back into
  `Model.loadAdapter` for inference.
- **GeGLU backward clamp** — `geglu_backward.wgsl` was missing the
  [-10, 10] tanh-input clamp the forward got; surfaced as all-NaN
  gradients at layer 33+ of the backward pass on Metal. Fix matches
  the forward formula bit-identically.
- **`overfit_one_smoke` + `per_position_smoke` integration tests** —
  CI now gates on a 200-step overfit (17.72 → 0.00 loss) and a 3-step
  PerPosition micro-batch (89 % loss drop) so backward regressions
  fail loud.
- **`eval_adapter` example** — loads a trained safetensors blob,
  decodes it back through `InferenceAdapter`, and runs a generation
  smoke against a real GGUF.

### Inference engine (`rullama`)

- **Vision encode is now one CommandEncoder per ViT block** + a GPU
  fence between blocks (`queue.on_submitted_work_done` + `device.poll`
  + oneshot). The per-layer progress callback now fires at real GPU
  pace rather than at record-time, killing the "frozen at 16/16"
  warm-cache freeze. Matches the text path's M7 pattern and CLAUDE.md's
  "one CommandEncoder per transformer layer" rule.

### Example PWAs

- **Fine-tune tab.** New first-class panel that owns a `TrainingSession`,
  runs in the foreground over the loaded model, and writes
  the resulting safetensors blob to OPFS. Adapter list, train/eval
  controls, loss chart, and one-click `Model.loadAdapter` are wired in.
- **Progress strip spans encode + splice + prefill.** Previously the
  strip vanished the moment `encodeImage()` resolved, leaving the user
  staring at 2-3 min of silence while the JS loop ran ~256
  `stepWithEmbedding` calls per image. The strip now carries a
  `phase` discriminator (`encoding` / `embedding` / `prefill`) and
  ticks through every phase, only clearing when the gen loop starts.
- **Service-worker swap reload (both mid-session and boot-time).**
  `clientsClaim: true` hands the new SW to live tabs silently; the
  old JS bundle still references hashed asset URLs the new precache
  no longer has, so any worker spawned after the swap 404s and
  surfaces as "checking OPFS…" hanging until the user hard-reloads.
  We now reload on `controllerchange` both during boot
  (inside `ensureFreshServiceWorker`) and after (`installPostBootSwReloadListener`).
- **`ModelLoader` polish** — kept the controls on a single row with
  status on its own line; dropped the redundant refresh-list button.

### Tooling / deploy

- `Dockerfile` builds the wasm bundle from `rullama-finetune` so the
  shipped `pkg/` exports both the inference (`Model`) and training
  (`TrainingSession`) wasm-bindgen surfaces.
- `cargo bump <version>` (`xtask`) updates both crate manifests AND
  the `rullama = { path, version = "MAJOR.MINOR" }` constraint in
  `rullama-finetune` so the next `cargo publish` resolves cleanly.
- `scripts/publish.sh` orchestrates the two-stage publish (rullama
  → wait for crates.io index → rullama-finetune).

## [0.2.0] — 2026-05-17

### Public API (semver-covered modules)

- `api::Model` — additive: `cancel_multimodal_encode_native()` (+ JS
  `cancelMultimodalEncode`), `release_vision_weights_native()`,
  `release_audio_weights_native()`, `cached_weight_bytes_native()`,
  `save_kv_state_native()` (+ JS `saveKvState`),
  `restore_kv_state_native()` (+ JS `restoreKvState`),
  `render_chat_for_continuation_native()`.
- `sampling::Sampler` — additive: `dump_state()` / `load_state()` for
  RNG + options + history snapshotting.
- `error::RullamaError` — **breaking**: new `Cancelled` variant. The
  enum is not `#[non_exhaustive]`; downstream exhaustive matches need
  to be updated. This is why this release is `0.2.0` rather than a
  patch.

### Inference engine

- KV cache + sampler suspend/resume — single byte blob (`RLMS` magic,
  versioned) that round-trips position, history, RNG cursor, and 26
  GPU layer pairs. Enables continuation of a partial assistant turn
  after the runtime is killed.
- Continuation chat template — re-renders a conversation so the model
  keeps writing the last assistant turn rather than starting a new one.
- Multimodal cooperative cancel — vision/audio encoders check a shared
  flag between transformer layers and bail with `RullamaError::Cancelled`.
- Multimodal encode progress callback + cached weights + chained
  encoder for vision and audio (matches the text-path M7 pattern).
- Weight-cache eviction by prefix — explicit hooks so the iPhone path
  can release vision/audio weights between turns.
- GeGLU uses tanh approximation, matching ggml's `geglu_split` (text
  parity tightening).

### Example PWAs

- Settings sidebar split into General / Sampling / Voice tabs (the
  Voice tab is disabled with a tooltip on text/vision-only models).
- Voice (VAD) capture wired through the audio tower, with adjustable
  silence cutoff / threshold / pre-roll / min-frames / max-record.
- Mid-generation auto-suspend on iOS backgrounding, with auto-resume
  on next boot — covers the live-tab recovery, pre-encode resume,
  and cross-tab race paths. Multimodal generations survive an iOS
  kill via OPFS-persisted pixels / PCM.
- Auto-resume of interrupted GGUF downloads on next boot.
- Screen wake lock held during download + generation so iOS doesn't
  sleep the device mid-token.
- Self-healing boot watchdog + "Reset app data" escape hatch for
  stuck PWA states.
- Service-worker render gating: drops the update-dialog round-trip;
  the page renders against fresh assets directly.
- `hasVision` / `hasAudio` primed from the catalog so the mic /
  image buttons appear at `modelStatus=ready` instead of after the
  first probe.
- Loader badge shows percent done, not percent remaining.
- `Defaults` button uses `Undo2` (revert-style) instead of `RotateCcw`
  (refresh-style) to disambiguate it from the `Reset app data` button.

### Tooling / deploy

- `cargo docker:*` aliases dispatched through a tiny std-only `xtask`
  binary (`docker:build` / `start` / `stop` / `restart` / `logs` / `ps`).
- Dockerfile now `COPY`s `xtask/` so the workspace parses cleanly
  during the wasm-pack build inside the container.
- R2 CORS allowlist no longer includes `localhost` origins.

### Docs

- `CLAUDE.md` at the repo root — workspace layout, architectural
  invariants (Worker isolation, per-tile fetches, one encoder per
  layer), and the small-public-API rule.
- README section for the docker/deploy flow and `cargo docker:*`
  aliases.

## [0.1.0] — 2026-05-14

Initial public release. Two crates published to crates.io:

- **`rullama`** — browser-resident Gemma 4 inference; pure Rust → WebAssembly + WebGPU.
- **`rullama-finetune`** — native local LoRA fine-tuning on the same wgpu kernels (experimental).

### Inference engine (`rullama`)

- Gemma 4 text inference (`gemma4:e2b`, `gemma4:e4b`) on desktop browsers and iPhone.
- Vision (ViT) + audio (Conformer) multimodal towers, bit-identical to Ollama on desktop.
- GGUF v3 loader with `Q4_K`, `Q6_K`, `F16`, `F32` quants — the full Ollama `Q4_K_M` mix.
- Streaming load via HTTP byte-range or OPFS `FileSystemSyncAccessHandle`; wasm peak stays in tens of MiB regardless of model size.
- Chained `CommandEncoder` + per-layer submits keep the GPU busy on tight-RAM phones.
- GPU-resident KV cache, multi-turn chat, mid-generation stop, persistent history.
- ~70 hand-written WGSL kernels — `matmul` (Q4_K / Q6_K / F16), `rmsnorm`, `rope_neox`, `geglu`, `attention` (incl. HPD-f16, block-local, subgroup variants), `conv2d`, plus the backward set for training.
- Public API restricted to `api::Model` / `api::ChatMessage` / `api::ChatRole` / `api::GenerateOptions`, `error::RullamaError` / `error::Result`, and `sampling::SamplingOptions` / `sampling::Sampler`. All other modules are `#[doc(hidden)]` implementation detail.

### Fine-tuning (`rullama-finetune`)

- Native-only LoRA SGD over the same wgpu kernel set — no Burn, no PyTorch.
- Rank-r LoRA on `attn_q` / `attn_k` / `attn_v` / `attn_o` and the FFN projections.
- Adam optimizer, global L2 gradient clipping, gradient accumulation, mixed precision, gradient checkpointing, warmup + linear / cosine / cosine-warm-restarts schedules.
- Cross-entropy loss (NextToken + PerPosition single-forward variant).
- Acceptance: 200-step overfit-one drops loss from `log(vocab_size) ≈ 12.5` → 0 on the dev fixture.
- Safetensors adapter save/load round-trips. On-disk checkpoints and inference hand-off back to the wasm runtime are deferred.

### Example PWAs

- `examples/web/` — React + Vite + Tailwind + Workbox; production-quality chat PWA with OPFS-backed model cache, conversation history in SQLite (via `rsqlite-wasm`), and service-worker-driven update dialog.
- `examples/pwa/` — vanilla JS bench harness and `safaridriver`-driven scripted iPhone runs.

### Known gaps

- iPhone multimodal: vision/audio towers skipped on the mobile loader to fit shared RAM.
- Matmul kernels are still naive; ≥10 tok/s on Mac requires the deferred tiled-matmul + bind-group caching + fusion pass (the M8 line on the roadmap).
- Greedy text output is *not* bit-identical to Ollama on every prompt — diverges on out-of-distribution inputs (e.g. "Once upon a time, there was a"). CPU↔GPU consistency within rullama is clean (≤8e-5 max abs).
- MoE `gemma4:26b` / `gemma4:31b` and non-Gemma architectures are out of scope.

[Unreleased]: https://github.com/Brainwires/rullama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Brainwires/rullama/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Brainwires/rullama/releases/tag/v0.1.0
