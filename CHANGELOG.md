# Changelog

All notable changes to **rullama** are tracked here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays in the 0.x series, only the modules listed in the
[stability section of `lib.rs`](crates/rullama/src/lib.rs) (`api`, `error`,
`sampling`) are covered by semver. Everything else is `#[doc(hidden)]` and
may move in any patch release.

## [Unreleased]

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
