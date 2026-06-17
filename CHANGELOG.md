# Changelog

All notable changes to **rullama** are tracked here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version stays in the 0.x series, only the modules listed in the
[stability section of `lib.rs`](crates/rullama/src/lib.rs) (`api`, `error`,
`sampling`, `lora`) are covered by semver. Everything else is `#[doc(hidden)]`
and may move in any patch release.

## [Unreleased]

A broad surface expansion since 0.4.0. The Gemma 4 family grows from two
dense models to the full runnable GGUF lineup — three new weight quants
(Q4_0 QAT, Q8_0, Q5_0), the 12B architecture, and the **26B-A4B sparse
MoE** with per-expert weight streaming that fits it on a low-VRAM GPU. A
validated **DiffusionGemma** (block-diffusion text) CPU forward + GPU
kernel set lands as a preview. Beyond text generation: an **embeddings +
RAG** stack (EmbeddingGemma), function-call **tool rendering + LoRA**, and
a second TTS engine — **StyleTTS2-LibriTTS** zero-shot voice cloning
alongside the Kokoro presets. Multi-turn chat gets **KV-cache reuse +
prompt caching**: the system prompt is pre-warmed once and conversations
reuse their resident (and OPFS-persisted) KV cache, so a new turn prefills
only the new message instead of re-reading the whole chain.

### Public API (semver-covered modules)

No changes to `api`, `error`, `sampling`, `lora`. New sibling classes
(`EmbeddingModel`, `StyleTtsClone`, and an in-flight DiffusionGemma
surface) are their own entry points, not part of the four covered
modules. Everything below is internal engine work, the new model
surfaces, or PWA / tooling assets.

### Gemma 4 — weight quants & model coverage

- **Q4_0** — runs Google's QAT (quantization-aware-trained) builds:
  `gemma4:{e2b,e4b,12b}-it-qat`. The forward was un-hardcoded from Q4_K to
  a dtype-routed `matmul_quant_chained` (Q4_K / Q6_K / Q4_0 / F16). QAT is
  ~half the download at preserved quality. Inference-only — fine-tuning on
  a QAT base fails early with an actionable error (train Q4_K_M, deploy QAT).
- **Q8_0** — 8-bit (`-it-q8_0` tags), the highest-quality runnable quant;
  byte-exact vs ggml `dequantize_row_q8_0`, byte-addressed fused
  dequant-matmul + GPU-vs-CPU parity.
- **Q5_0** — 5-bit, required for the 26B / DiffusionGemma Q4_K_M mix (Q5_0
  on ~half the `ffn_down_exps`). Caught by the real 26B GPU run, not the
  synthetic tests.
- **12B architecture** — per-layer KV-head arrays (8 SWA / 1 global),
  no-V global attention (V := the raw K projection), and 4-byte GPU-upload
  alignment for odd-multiple Q6_K rows. Un-breaks `gemma4:12b` and enables
  the 12B QAT build.

### Sparse MoE — `gemma4:26b-a4b` (128 experts, top-8)

- The MoE FFN runs IN PARALLEL with the dense MLP per layer, mirroring
  Ollama's `model_text.go` 1:1: router (unweighted rmsnorm → ×1/√d →
  ×scale → softmax → top-k → renorm) → fused `ffn_gate_up_exps` → GeGLU →
  `ffn_down_exps` → weighted combine, under `post_ffw_norm_1/2`. A
  per-layer branch, not a new model family (it's still `general.architecture
  = gemma4`); the standing "MoE out of scope" line is retired.
- CPU oracle + GPU kernels — `moe_router`, MulmatID `moe_expert_matmul`
  (Q4_K / Q5_0 / Q8_0), `moe_geglu_halves`, `moe_combine`, plus batched
  variants for the diffusion canvas — each GPU-vs-CPU parity-tested
  (≤1.1e-4 on real blk.0/5/29 weights). New 3-D expert-slice dequant +
  native `FileFetcher` (stream blobs bigger than RAM). `TrainingSession`
  rejects MoE bases (inference-only).
- Load-bearing GPU fix surfaced here: **Iris/Metal silently runs only 64
  of a 128-thread workgroup** — WG=64 is now a hard rule, and parity tests
  exercise the full index range.

### Low-VRAM weight streaming

- The MeBP per-layer weight-destroy pattern extended to **inference + MoE**:
  the 26B GPU forward runs on a 16 GB Mac at **~0.6 GB peak** (27× under
  its 16.7 GB resident set), output byte-identical to the CPU oracle — the
  model didn't fit at all before.
- **Per-expert streaming** — fetch only the routed top-8 of 128 experts
  per layer (`buffer_expert_async` range-fetch), ~16× less bandwidth; with
  the non-expert weights kept resident, the 26B runs at ~4 s/tok (down from
  ~8) at ~1.6 GB peak, output unchanged. (`moe_stream_smoke` example;
  `DG_LAYER_TIME` phase timing.)

### DiffusionGemma — block-diffusion text (preview)

- Validated CPU canvas-forward for the `diffusion-gemma` architecture (the
  26B-A4B MoE backbone run non-autoregressively over a 256-token "canvas"):
  the entropy-bound sampler, region-aware unified mask (bidirectional
  canvas / causal-windowed prompt), full-sequence masked attention, and
  the self-conditioning gated MLP — all mirroring llama.cpp PR 24423 and
  diffed against its `llama-diffusion-gemma-eval` oracle (98.4% argmax;
  99.6% with self-conditioning; a per-layer bisection confirms layer-0
  correlation 0.9998, i.e. the math is correct and the residual drift is
  inherent MoE routing-boundary accumulation).
- **Full GPU canvas forward** (`reference/diffusion/gpu.rs`): the entire
  forward — dense projections AND the 128-expert MoE FFN — now runs on the
  GPU, batched over all canvas positions. Dense Q4_K/Q5_0/Q8_0 matmuls reuse
  the batched MoE-expert kernel with `top_k = 1` (a batched dense quant
  matmul for free); Q6_K (`attn_v`, the tied lm_head) falls back to a
  single-row loop; the MoE runs the batched router / expert-matmul / GeGLU /
  combine path with per-LAYER expert streaming (each layer's ~0.5 GB of
  stacked experts is made resident then destroyed). Validated GPU-vs-CPU on
  the real 16.8 GB streamed model: argmax-exact, max-abs ≤ 0.0004.
- **`DiffusionGemma` engine + browser surface**: a sibling wasm class
  (`src/diffusion.rs`) like `EmbeddingModel` — streaming loader + the
  entropy-bound denoise loop over the GPU forward. Native generation
  validated end-to-end ("The capital of France is" → " Paris."). The wasm
  `denoiseStep` surface (JS drives the loop, rendering the canvas condensing
  out of noise) is wired into the PWA: `diffusiongemma:26b-a4b` is now
  selectable, streams from OPFS, and generates via a denoise-loop chat path.
  Published to R2 (16.8 GB Q4_K_M). Desktop-class; tens of seconds per
  denoise step on weak GPUs.

### Embeddings + RAG

- **EmbeddingGemma-300M** (`gemma3` arch, encoder-only) → the
  `EmbeddingModel` sibling class over a bidirectional GPU forward,
  bit-identical to the CPU oracle (cosine 0.9997 vs Ollama).
  SentencePiece-unigram tokenizer (`tokenizer::spm`). Powers the PWA
  Knowledge tab (drop/paste docs → chunk → embed → rsqlite-wasm vector
  store) and per/cross-conversation chat RAG. Streaming loader keeps the
  621 MB GGUF from ever being fully resident (iPhone-safe).

### Chat — tool calling

- `<tool_call>{json}</tool_call>` renderer rendered as a structured block,
  with a tolerant parser hardened for real-model output (e.g. a missing
  `>` on the open tag). Function-call LoRA recipe + a canonical
  dataset/generator. Training made interruptible + resumable
  (checkpoint/restore) for slow GPUs.

### Chat — KV-cache reuse & prompt caching

Multi-turn chat no longer re-reads the whole conversation on every send.
The GPU KV cache stays resident across turns and the engine prefills only
what's new — the "Reading prompt N/total" phase now scales with the new
message, not the chain length.

- **Cross-turn prefix reuse** — the inference core tracks the exact token
  sequence resident in the KV cache; a new turn feeds only the suffix past
  the longest matching prefix instead of resetting and re-prefilling from
  `<bos>`. Lives in the (cross-tab-shared) core, so it can't go stale
  behind another tab. Correctness is gated purely on a token-content
  match — any mismatch (edited history, changed system prompt, model swap)
  safely falls back to a full prefill, so the worst case is "no speedup,"
  never wrong output.
- **`Forward::truncate_kv`** (new) — drops KV positions ≥ N and rewinds
  `pos`, enabling *longest-common-prefix* reuse: a brand-new chat (or a
  chat switch, or editing an earlier message) keeps the shared
  system-prompt head and re-prefills only the divergent tail. Sound for
  every layer type — the cache is linear and sliding-window attention is a
  compute-time mask, not a ring buffer — so it's pure `pos`/`kv_lens`
  bookkeeping, no GPU work. Parity-verified bit-identical to a fresh
  prefill of the kept prefix (max-abs logit diff 0.0).
- **System-prompt pre-warm** — after a model loads, a new "preparing"
  phase prefills the system block into the KV cache (shown as a second
  segment in the load progress bar / boot splash) so even the FIRST chat
  hot-starts. Re-warms when the system prompt is saved or thinking /
  tool-mode toggles. `buildSysContent` orders the static system core ahead
  of dynamic RAG / GPS content so the warmed prefix is always a clean,
  reusable prefix of a real turn. The warm is **persisted to OPFS, one
  file per model digest** (keyed by the model + system-prompt signature):
  on reload it's restored instead of recomputed, so the same system prompt
  is never prefilled more than once per model — even across page reloads.
  (No-adapter only: a LoRA changes the cached K/V and is applied after the
  load-time warm, so adapter sessions recompute rather than risk a stale
  restore.)
- **Per-conversation KV snapshots** — a conversation's KV cache is
  persisted to OPFS (an `RLCV` envelope = resident token ids + the KV /
  sampler blob, model-digest tagged, LRU-capped, size- and quota-guarded)
  and restored on reopen, so reloading the page and reopening a long chat
  skips the re-prefill entirely. Composes with prefix reuse — a stale
  snapshot just becomes a shorter reusable prefix.
- **Per-turn date/time** — each user turn carries a frozen `[date time]`
  prefix so the model always knows the current time, done cache-safely:
  the stamp is fixed per message (re-rendered identically from its
  `created_at`), so it rides the always-re-fed user turn without shifting
  the cached prefix. A static system note teaches the model to read it.
- **UI** — a dedicated "Loading model" view (spinner + progress + Stop)
  replaces the picker while a model is loading/preparing, and the
  model-tab controls lock during load. The system prompt is now a
  read-only display with Edit → Save / Cancel and a "preparing…" indicator
  while the new prompt warms.

### Speech engine (TTS + voice cloning)

The Voice tab gains a second engine — **StyleTTS2-LibriTTS** zero-shot
cloning (desktop-only) alongside the existing Kokoro preset voices — with
GPU voice-creation, GPU style-diffusion prosody, and the full English
Kokoro voicepack set.

- **GPU style encoder** — voice *creation* (reference clip → 256-d style)
  now runs on the GPU, not just synthesis. New channel-first kernels
  `conv2d_chf` / `avg_pool2d_half_chf`; bit-exact vs PyTorch (max-abs 5.5e-7).
- **Style-diffusion prosody (α=0.3/β=0.7)** — restores natural,
  text-appropriate prosody (the prior α=β=0 path was flat / "accented").
  StyleTransformer1d denoiser + KDiffusion `denoise_fn` (σ_data=0.2) +
  ADPM2 sampler + Karras schedule. CPU oracle bit-exact (5e-6); GPU
  denoiser (f16 matmul + flash attention + AdaLayerNorm + new exact-GELU
  kernel) is **7–17× faster** than CPU (0.7–1.2 s), corr 0.97 vs PyTorch.
  Cloned synthesis uses it by default.
- **Bind-cache leak fixed** — `StyleTtsGpu` now evicts its per-call scratch
  buffers from the shared bind-group cache on `Drop`. Previously every
  synth/encode leaked descriptor-table entries, so a long clone session
  grew unbounded until the GPU exhausted (a tight-loop bench hard-locked a
  weak integrated GPU after ~3–5 min). Real production fix.
- Mel frontend (`compute_style` filterbank + window) baked into the GGUF
  so the Rust encoder reads it directly. New `gpu_yield` dev hook
  (`ST2_GPU_THROTTLE_MS`, native-only, no-op in prod) lets a weak GPU
  yield between stages for batch dev/bench runs.

### Cloning surface (desktop-only)

- `StyleTtsClone` wasm-bindgen API: async `load` / `encodeVoice` /
  `synthesize`, all GPU. Loads off the iPhone text-only path (never on
  mobile). Model is f32-with-diffusion, **543 MB**, on R2.

### Models / distribution

- **Gemma 4 catalog** — published to R2 (`models.brainwires.dev`) and
  added across the three catalog mirrors (`web/src/lib/api.ts`,
  `web/server/ollama.ts`, `docker/entrypoint.sh`): the QAT trio
  `gemma4:{e2b,e4b,12b}-it-qat` (Q4_0), the Q8_0 trio
  `gemma4:{e2b,e4b,12b}-it-q8_0` (e2b/e4b full multimodal, 12b text-only),
  the standard `gemma4:12b`, and the `gemma4:26b` MoE (heavy ⚠). MLX
  (`-mlx`/`-mxfp8`/`-nvfp4`) and `-cloud` tags are explicitly out — not
  GGUF / server-side.
- **EmbeddingGemma-300M** — 621 MB GGUF on R2, in the Knowledge-tab catalog.
- **Kokoro: all 28 English voicepacks** (af/am American, bf/bm British,
  ♀/♂) shipped as Voice-tab presets — the GGUF previously bundled only
  `af_heart`. New f16 GGUF (170.8 MB) on R2; catalog `digest`/`size` bumped
  (one-time OPFS re-download). StyleTTS2-LibriTTS cloning GGUF added to the
  catalog.

### Tooling

- New examples: `moe_parity` / `moe_layer_parity` / `moe_chained_smoke`
  (26B MoE oracle + GPU layer parity), `moe_stream_smoke` (low-VRAM
  streaming), `diffusion_parity` / `diffusion_config_probe` (DiffusionGemma
  vs the llama.cpp oracle), `embed_parity`. Test fixtures now read GGUF
  **headers only** (no whole-blob `fs::read`) — fixed a lib-suite OOM.
- `clone_fidelity_harness` example — A0 calibration that isolates
  reference-quality from speaker by cloning Kokoro `af_heart`'s own clean
  output and measuring speaker-similarity. Finding: clean clone 0.96
  (ceiling 0.97) vs a noisy clip 0.46 (below the 0.70 different-speaker
  floor) — clone quality is dominated by reference cleanliness, and
  quantity saturates by ~15 s.
- `convert-styletts2-gguf.py` (+ diffusion weights, mel bake) and
  `styletts2_dump_diffusion_fixtures.py`; Kokoro converter now bundles all
  downloaded voicepacks.

## [0.4.0] — 2026-06-01

Mac fast path + dev-server overhaul.

The iPhone-targeted code path that landed in 0.3.x trades wall-clock for
GPU heap headroom — fine when the heap is the bottleneck, very much not
fine on a 32 GB MacBook with no heap pressure. 0.4.0 puts that path
behind a `Memory-tight` flag (default `false`) and adds a small native
Rust dev server that finally retires the two Python `http.server`
scripts the repo has been carrying.

### Public API (semver-covered modules)

No changes to `api`, `error`, `sampling`, `lora`. Everything below is
either internal (the engine path that finetune drives) or the PWA / dev
tooling.

### Engine (rullama)

- New `Forward::mobile_mode` flag gates **eight** previously-unconditional
  iPhone workarounds (`mobile_mode = false` on Mac is the fast path):
  - per-layer `drop_blk_layer_range_destroy` between forward layers
  - per-layer `drop_prefix_destroy("blk.{i}.")` between backward layers
  - early `drop_prefix_destroy("token_embd")` after the head outproj
  - cross-step `drop_prefix_destroy("")` at end-of-step
  - chunked-destroy yields + per-layer epilogue yield
  - 0-ms JS yield between recompute submit and backward_layer
  - 0-ms JS yields between prefill tokens
  - tiled head_outproj over the vocab axis (8 tiles when mobile, 1 otherwise)
- Result on Mac: 1772 MiB resident through the entire backward walk,
  cross-step weight cache survives → ~25% wall-clock reduction on the
  Mac CDP harness vs the 0.3.x iPhone-tuned path.

### Fine-tune (rullama-finetune)

- `TrainingHyperparams::memory_tight: bool` (serde-default `false`) —
  what the new `mobile_mode` reads. PWA's `Memory-tight (iPhone-safe)
  preset` toggle plumbs through `trainingStart`'s hparams JSON.
- `mem_tight_repro` example explicitly sets `memory_tight: true` so the
  native canonical run still exercises the iPhone path. Bit-identical:
  9.0703 / 11.0554 / 9.1028 across 3 epochs.

### PWA (web)

- Fine-tune tab — `Memory-tight` toggle moved to the bottom of the
  settings stack and labelled **Highly experimental** (slower on Mac,
  currently crashes mid-step on iPhone 16e). Default state is off on
  devices with `navigator.deviceMemory ≥ 4 GB` and non-iOS UA.
- New device-capability gate (`useTrainingCapability`) blocks the
  Fine-tune tab on iOS UA, missing WebGPU, `maxBufferSize < 512 MB`, or
  `deviceMemory < 4 GB`, rendering a `TrainingBlockedScreen` with a
  clear reason instead of a 30-second wait into a crash.
- `lib/api.ts::blobUrl(m)` — `?localBlob=PORT` now wins over the baked-in
  CDN URL. Before this, the override was silently ignored for any model
  in `BAKED_IN_MODELS` — the very case the override exists for.
- Crash-detect — the false "Last session ended unexpectedly" toast on
  every refresh is gone. Mount handler now consults BOTH the manifest's
  `cleanExit` flag AND the pagehide-written `localStorage` marker; only
  flags when both are absent. The pagehide handler was also fixed to
  cache the worker session id synchronously into a `useRef` so it can
  write the marker without an async RPC (which the tab teardown beats).
- Defaults realigned to the verified garlic-acceptance recipe: initial
  `stepsBudget` 32 → 50, default `repetition_penalty` 1.1 → 1.3.
- shadcn `AlertDialog` + `useConfirm()` hook replace the old
  `window.confirm` for the >200 MB download prompt — works under
  Playwright/CDP automation, which the native dialog didn't.

### Dev tooling

- New crate `crates/rullama-devserver/` — native Rust dev server.
  Replaces `web/serve-iphone.sh` (LAN HTTPS) and
  `web/serve-tunnel.sh` (HTTP behind Cloudflare tunnel).
  - `cargo dev` brings up the full stack: axum on `:25321` +
    Vite child on `:5173` (with HMR WebSocket forwarding so editing
    React works through either port) + fs watcher on
    `crates/{rullama,rullama-finetune}/src/**` that auto-runs
    `wasm-pack build` and broadcasts a `wasm-rebuilt` event over WS,
    triggering a page reload in `web/src/lib/dev-hmr.ts`.
  - `cargo dev -- --public` composes tunnel-safe defaults: serves
    `web/dist/` instead of reverse-proxying Vite (Vite's
    `fs.allow=[repoRoot]` would otherwise leak the entire repo), disables
    `/api/log` writes, disables `/api/models` listing, disables
    `/__rullama-dev-ws`.
  - Endpoints are wire-identical to `serve-tunnel.sh`: GET/HEAD on
    `/api/models`, GET/HEAD on `/api/blob/{family}:{tag}` with 1 MiB
    range streaming, POST `/api/log` (8 KiB body cap), OPTIONS preflight.
  - COOP / COEP / per-route CORP headers on every response; CORS is
    allow-list only (no wildcard), driven by `--cors-origins`.
  - Path-traversal hardened via canonicalization + base-dir prefix check
    on `/pkg/*` and the `dist/` fallback.
  - 18 integration tests in `tests/http_endpoints.rs` covering every
    route shape, security header, public-mode disabling, and the
    8 KiB body cap (`cargo test --manifest-path
    crates/rullama-devserver/Cargo.toml --release`).
- PM2 ops at `ops/pm2/`:
  - `ecosystem.config.cjs` — runs `rullama-devserver --public --cors-origins
    https://rullama.brainwires.net` directly off the release binary
    (NOT through `cargo run`).
  - `setup.sh` — idempotent bring-up: builds binary + dist, restarts
    the PM2 entry, saves the process list.
  - `README-for-next-agent.md` — hand-off note describing the two
    modes, the security boundary, the right way to add a new route.
- `cargo dev` is an xtask alias (`.cargo/config.toml`) dispatching to
  `cargo run --manifest-path crates/rullama-devserver/Cargo.toml --release`.
  The crate is **excluded** from the workspace so
  `cargo build --workspace --target wasm32-unknown-unknown` doesn't try
  to compile axum / notify / tokio for wasm.
- Mac CDP automation harness (`web/test/mac-cdp-test.mjs`) —
  direct Chrome DevTools Protocol harness bypassing the Playwright
  React 18 click bug, with dataset save+reuse, explicit Memory-tight
  uncheck before training, and post-click worker-beacon verification
  that `memory_tight=false` actually reached the Rust side.

### Migration notes

- `TrainingHyperparams::memory_tight` is `serde(default = false)`, so
  older JSON payloads keep working — but they will now run the
  **Mac fast path** instead of the iPhone code path. If you were
  depending on the per-layer destroy behavior on a memory-tight device,
  pass `memory_tight: true` explicitly.
- `serve-iphone.sh` is left in place for the (paused) iPhone path; it
  is **NOT** deprecated. `serve-tunnel.sh` is superseded by
  `cargo dev -- --public` and can be removed in a future patch release.

## [0.3.0] — 2026-05-19

Two flagship landings:

- **In-browser fine-tuning.** Same `Model` you've been running inference
  on can now consume a LoRA adapter the PWA trained in the foreground
  tab — no native build, no separate inference engine, no round-trip
  through disk.
- **Multimodal works on iPhone.** Both vision and audio towers run
  end-to-end on A18 GPU through iOS Safari WebGPU after the per-block
  encoder + GPU fence rework, with the towers' GPU residency dropped
  between phases so the WebContent process doesn't OOM during prefill.

Plus a progress-strip overhaul, an in-engine speech-to-text path, and
a load of PWA reliability fixes.

Bumped to `0.3.0` for the new `lora` module + the scope of the
fine-tune and multimodal-on-iPhone surfaces. The existing public API is
preserved, with one minor signature change called out below.

### Public API (semver-covered modules)

- New `lora` module — `InferenceAdapter::parse_safetensors(bytes)` decodes
  the safetensors blob `TrainingSession::saveAdapter()` writes. `Model`
  owns the active adapter; consumers don't construct it directly.
- `api::Model` — additive:
  `has_adapter_native()` (+ JS `hasAdapter`),
  `adapter_slot_count_native()`,
  `load_adapter_native(bytes)` (+ JS `loadAdapter`),
  `clear_adapter_native()` (+ JS `clearAdapter`),
  `load_streaming_with_max_context(...)` and
  `load_from_opfs_native(..., max_context)` (+ JS `loadFromOpfs` now
  takes an optional max-context argument).
- **Minor signature change**: `release_vision_weights_native` /
  `release_audio_weights_native` now take `&mut self` (was `&self`).
  Callers that store `Model` behind a non-`mut` binding need to flip it
  to `let mut`. This is the only breaking shape change in the release.
- `error`, `sampling` — unchanged.

### Inference engine (`rullama`)

- **Per-encoder GPU fence shared by vision + audio.** `vision::encode`
  and `audio_gpu::encode` both submit one `CommandEncoder` per
  transformer block, then await a GPU fence (`queue.on_submitted_work_done`
  + `device.poll(Wait)` + oneshot) before the next block. The fence
  helper lives in `backend::dispatch::fence_submitted_work` so both
  towers share it. This restores CLAUDE.md's "one CommandEncoder per
  transformer layer" rule and is what unblocks iPhone multimodal —
  iOS Safari WebGPU was silently corrupting state on the previous
  single-encoder-spanning-all-12-blocks audio path, surfacing as a
  worker crash on the first `step()` after `encodeAudio`.
- **`loadFromOpfs` honors `max_context`.** Was previously hardcoded to
  4096 on the multimodal load path; the JS layer's intent (e.g. iPhone
  wanting 2048) was silently dropped. Now mirrors the text-only
  loader's caller-supplied cap, saving ~600 MB of KV pre-allocation
  on a 2048-cap mobile load.
- **Soft tokens see the LoRA adapter.** `step_with_embedding_native`
  now routes through the same adapter branch as `step_native`, so an
  adapter-loaded `Model` actually applies the adapter to every image /
  audio soft token spliced into the prompt.
- **GPU fault notification.** Inference paths now signal a `gpuFault`
  event when wgpu surfaces a device-lost / OOM so the PWA can
  distinguish a real GPU failure from a hung worker, instead of
  silently timing out.
- **Multimodal scratch is now ephemeral.** `release_vision_weights` and
  `release_audio_weights` drop the `VisionForward` / `AudioForward`
  structs alongside the cached weights. Next encode rebuilds them
  lazily from the cached GGUF reader. ~3 GB of GPU memory each that
  was previously stuck resident until the page reloaded.

### Fine-tuning (`rullama-finetune`)

- **wasm32 port.** The crate now compiles to wasm32-unknown-unknown
  (was native-only). On wasm builds the whole crate is gated off so it
  remains empty; on wasm-bindgen builds (`crate-type = ["cdylib"]`) it
  exposes a `TrainingSession` surface.
- **`TrainingSession` wasm-bindgen surface.** JS callers can run a
  forward → loss → backward → Adam step entirely in the foreground tab
  against the same wgpu kernels the inference path uses. Save trained
  weights as a safetensors `Uint8Array` and load them back into
  `Model.loadAdapter` for inference.
- **Real gradient checkpointing.** A shared per-step activation scratch
  replaces the per-layer captures when `gradient_checkpointing=true`;
  the backward walker replays each layer's forward from its saved
  `hidden_in` before consuming it. Cuts activation memory ~`n_layers×`
  in exchange for one extra forward per backward step.
- **`TrainingSession::probe(model, lora_cfg, hp)`** — trial-allocates
  the scratch + LoRA buffers against a borrowed `Model` so the UI can
  refuse a `new(...)` call that would otherwise consume the Model and
  then fail mid-allocation. Returns the estimated GPU bytes on
  success, `Err` with reason on failure.
- **Per-layer training progress beacons.** Native + wasm32 backward
  passes fire a callback with phase / layer / total on every layer
  boundary; the Fine-tune tab uses them to drive a live progress strip
  rather than a frozen spinner.
- **Per-layer encoder cancellation.** Training honors a cooperative
  cancel flag checked between layers, mirroring the multimodal-encode
  cancel — bail with a clean error mid-step instead of hanging the
  worker.
- **`checkpoint_parity` integration test** — gates that the gradient
  checkpointing path produces gradients within tolerance of the
  reference no-checkpoint path.
- **GeGLU backward clamp** — `geglu_backward.wgsl` was missing the
  [-10, 10] tanh-input clamp the forward got; surfaced as all-NaN
  gradients at layer 33+ on Metal. Fix matches the forward formula
  bit-identically.
- **`overfit_one_smoke` + `per_position_smoke` integration tests** —
  CI now gates on a 200-step overfit (17.72 → 0.00 loss) and a 3-step
  PerPosition micro-batch (89 % loss drop).
- **`eval_adapter` example** — loads a trained safetensors blob,
  decodes it back through `InferenceAdapter`, and runs a generation
  smoke against a real GGUF.

### Example PWAs

- **Fine-tune tab.** New first-class panel that owns a `TrainingSession`,
  runs in the foreground over the loaded model, and writes the
  resulting safetensors blob to OPFS. Adapter list, train/eval
  controls, loss chart, GPU-byte budget hint, and one-click
  `Model.loadAdapter` are all wired in.
- **`PipelineProgress` strip spans encoding + embedding + prefill.**
  Renamed from `VisionProgress` to cover audio transcription too. The
  strip carries a `phase` discriminator and ticks through every phase
  so the user isn't staring at 2-3 min of silence between
  `encodeImage` / `encodeAudio` resolving and the gen loop starting.
- **Vision + audio tower weights released between encode and
  prefill / transcribe.** The PWA now drops the multimodal towers as
  soon as soft tokens are extracted, freeing ~3 GB each on iPhone
  before the text prefill kicks off. Re-uploaded on demand for the
  next attachment.
- **Mic = in-engine STT via rullama.** The mic button now always runs
  audio through `encodeAudio` + soft-token splice for a transcription
  pass instead of routing through a separate API. Paperclip icon
  replaces the Plus, and the paperclip accepts audio files for the
  same path.
- **`transcribeAudio` acquires its own session lock.** Audio
  transcription no longer races against an in-flight chat send.
- **NetworkFirst navigation handler.** Replaces the precache-based nav
  strategy that left post-deploy reloads stranded on stale chunk
  URLs. The mid-session and boot-time `controllerchange` reload
  workarounds are gone — every reload picks up live HTML
  referencing whatever chunk hashes the live deploy has right now.
- **Visible boot splash + graceful worker shutdown + auto-recovery
  on first stuck boot.** New `index.html` splash with a tiny
  `__rullamaBootStatus` global; worker shutdown handler explicitly
  releases vision + audio tower GPU resources; first-launch hangs
  recover automatically instead of black-screening.
- **OPFS write hardening.** Don't nuke the cached GGUF on a transient
  magic-check read failure; retry `FileSystemSyncAccessHandle` write
  acquisition while iOS GCs the previous lock; resume downloads on
  stream drop instead of restarting from byte 0.
- **`ModelLoader` polish** — controls on a single row with status on
  its own line; dropped the redundant refresh-list button.
- **Settings hints.** Multimodal models surface a hint about
  disabling "thinking" + temperature when sending image / audio
  attachments (gemma4's behavior on multimodal inputs is sharp
  enough that the default sampling assumes it).

### Tooling / deploy

- `Dockerfile` builds the wasm bundle from `rullama-finetune` so the
  shipped `pkg/` exports both the inference (`Model`) and training
  (`TrainingSession`) wasm-bindgen surfaces.
- `cargo bump <version>` (`xtask`) updates both crate manifests AND
  the `rullama = { path, version = "MAJOR.MINOR" }` constraint in
  `rullama-finetune` so the next `cargo publish` resolves cleanly.
- `scripts/publish.sh` orchestrates the two-stage publish (rullama
  → wait for crates.io index → rullama-finetune).
- `audio_parity` example handles mp3/m4a inputs via `afconvert`
  (macOS) or `ffmpeg` (Linux) so parity runs don't require a manual
  pre-conversion to wav.

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

- `web/` — React + Vite + Tailwind + Workbox; production-quality chat PWA with OPFS-backed model cache, conversation history in SQLite (via `rsqlite-wasm`), and service-worker-driven update dialog.
- `examples/pwa/` — vanilla JS bench harness and `safaridriver`-driven scripted iPhone runs.

### Known gaps

- iPhone multimodal: vision/audio towers skipped on the mobile loader to fit shared RAM.
- Matmul kernels are still naive; ≥10 tok/s on Mac requires the deferred tiled-matmul + bind-group caching + fusion pass (the M8 line on the roadmap).
- Greedy text output is *not* bit-identical to Ollama on every prompt — diverges on out-of-distribution inputs (e.g. "Once upon a time, there was a"). CPU↔GPU consistency within rullama is clean (≤8e-5 max abs).
- MoE `gemma4:26b` / `gemma4:31b` and non-Gemma architectures are out of scope.

[Unreleased]: https://github.com/Brainwires/rullama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Brainwires/rullama/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Brainwires/rullama/releases/tag/v0.1.0
