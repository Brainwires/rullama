# rullama

The **rullama apps** — the consumer product family, all running **Gemma 4 on your
local GPU** (with optional cloud providers). This repo holds three front-ends:

- **`apps/web/`** — a browser-resident AI **PWA** (React + Vite + Tailwind + Workbox)
  that runs inference **in the browser** via WebGPU + WASM. Chat (text / vision /
  audio-input), a Knowledge tab (drop docs → embed → RAG), a Fine-tune tab
  (in-browser LoRA), and a Voice tab — your data never has to leave the device.
- **`apps/native/`** — the **desktop + mobile** app (.NET / Avalonia) with a Rust
  `rust-core` C-ABI shim. Ships to the app stores.
- **`apps/cli/`** — the **agentic CLI** (BYOK providers, tools, MCP, sessions).

All three build on the **[rullama-framework](https://github.com/Brainwires/rullama-framework)**
platform — the OSS inference **engine** (`rullama-engine` + the `rullama-lora`
trainer) and agent **harness** (`rullama-*` crates: agents, tools, memory, RAG,
providers, MCP). The engine + harness live in that sibling repo, not here.

> **One brand — rullama — across the stack**, all at `rullama.com`.
> **Brainwires** is the company / GitHub org, not a project name. See the
> canonical topology doc:
> [`rullama-framework/docs/ARCHITECTURE-engine-harness.md`](https://github.com/Brainwires/rullama-framework/blob/main/docs/ARCHITECTURE-engine-harness.md).

## How each app builds on rullama-framework

| App | Consumes the platform via |
|-----|---------------------------|
| `apps/web/` | the engine's **wasm bundle** at `/pkg/rullama.js` (built from `rullama-framework/engine/rullama-lora`), driven in a Worker; cloud via the BYOK `/api/cloud/*` proxy. |
| `apps/native/` | the **C-ABI** `rust-core` shim linking `rullama-engine` + `rullama-lora` directly (P/Invoke — no HTTP, no wasm). |
| `apps/cli/` | **path-deps** the `rullama-framework` harness crates (its own Cargo workspace). |

The dev-server can also host the engine over an OpenAI-compatible
`/v1/chat/completions` endpoint for any local client.

## Layout

| Path | What it is |
|------|------------|
| `apps/web/` | The PWA (React + Vite + Tailwind + Workbox SW). Imports the engine wasm bundle from `/pkg/rullama.js`. |
| `apps/native/` | Desktop + mobile app: `app/` (.NET / Avalonia) + `rust-core/` (C-ABI cdylib over the engine). Own build; excluded from the root workspace. |
| `apps/cli/` | The agentic CLI — its own Cargo workspace (own `Cargo.lock`), path-deps `../../../rullama-framework`. Excluded from the root workspace. |
| `services/dev-server/` | The dev/serve HTTP server (Vite proxy, `/api/blob`, `/api/models`, `/api/cloud/*`, `/pkg/*`, cross-repo wasm-bundle watcher). Excluded from the workspace; run via `--manifest-path`. |
| `services/worker/` | Cloudflare Worker — the production BYOK cloud proxy (deployed standalone via `wrangler`). |
| `xtask/` | `cargo dev` + `cargo docker:*` dispatcher (std-only). The root workspace is just this. |
| `pkg/` | The engine wasm bundle (built artifact, `--out-name rullama`; gitignored, at the repo root). |

The inference engine + LoRA trainer live in the
[rullama-framework](https://github.com/Brainwires/rullama-framework) repo as
`rullama-engine` / `rullama-lora` (an isolated `engine/` wasm32 sub-workspace).
The iOS bench harness moved there too (`engine/tools/ios-bench`).

## `apps/web/` — what works today

- ✅ **`gemma4:e2b` text inference on the desktop** loads end-to-end and
  generates greedy output bit-identical to Ollama. (`gemma4:e4b` is
  shape-compatible — pull and try it.)
- ✅ **`gemma4:e2b` text inference on iPhone** — full Q4_K_M model loaded
  into iPhone 16e (A18, 8 GB shared RAM) and streaming tokens at ~4.65 tok/s
  via a Dedicated Worker + sync OPFS path. *Multimodal towers stay
  Mac-only for now; mobile picks the text-only loader (`max_context=512`).*
- ✅ **Vision + audio multimodal** on the desktop. ViT (16 blocks, 768
  hidden) + Conformer (12 blocks, 1024 hidden) towers run on the same wgpu
  device as the text path; soft tokens splice into the prompt via
  `<|image>` / `<|audio>` sentinels. Validated bit-identical to Ollama on
  a fixed image and a 30-second pangram WAV.
- ✅ **Q4_K + Q6_K + F16 + F32** quants (the actual mix in `gemma4:e2b` Q4_K_M).
- ✅ **Streaming load** via HTTP byte-range requests *or* OPFS sync access
  handles — the 7 GB GGUF never enters wasm linear memory in bulk. The
  PWA writes to OPFS once via `FileSystemSyncAccessHandle.write()` in a
  worker, and reads tile-by-tile during inference, so the wasm peak stays
  in the tens of MiB regardless of model size.
- ✅ **Multi-turn chat** with system prompt, mid-generation Stop, persistent
  KV cache.
- ✅ **Encoder chained + per-layer submits** (M7 + M15) — one CommandEncoder
  spans each transformer layer, submitted incrementally so the GPU drains
  smoothly even on tight-RAM phones.
- ✅ **In-browser LoRA fine-tuning (`rullama-lora`, wasm + native).**
  Backward kernels for matmul Q4_K / Q6_K, rmsnorm, rope, geglu, attention,
  cross-entropy; Adam optimizer over GPU buffers; rank-r LoRA on attention
  + FFN projections. 200-step overfit-one drops loss from ~17.7 → 0 on the
  dev fixture. Trained adapters export as safetensors and load back into
  the inference `Model` via `loadAdapter` — no roundtrip through native.
  The PWA's Fine-tune tab drives all of this in the foreground tab.
- ❌ MoE `gemma4:26b` / `gemma4:31b` — out of scope.
- ❌ Other architectures (llama, mistral, qwen, phi).
- 🛠️ **Mobile multimodal** — desktop multimodal works; the iPhone loader
  currently skips the vision/audio towers to fit in shared RAM. Lazy
  upload for those is a follow-up.

## `apps/web/` — running the browser PWA

You need:
- Rust ≥ 1.91 + `wasm-pack` (`cargo install wasm-pack --locked --version 0.13.1`)
- A WebGPU-capable browser (Chrome 113+, Edge 113+, recent Firefox; iOS
  Safari 17.4+ for phones)
- Ollama installed locally with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`)

### Get the wasm bundle (from the engine)

The engine bundle is built from a **sibling rullama-framework engine checkout**. With
`../rullama-framework/engine` present, `cargo dev` rebuilds it automatically
when engine source changes; otherwise build it once:

```sh
# Unified bundle — inference (`Model`) + training (`TrainingSession`) surfaces.
# Run inside the engine checkout. --out-name rullama keeps pkg/rullama.js stable.
# Override the location with RULLAMA_ENGINE_DIR.
cd ../rullama-framework/engine
wasm-pack build rullama-lora --target web --release \
    --out-dir ../../rullama/pkg --out-name rullama
```

This emits `pkg/rullama.js` + `pkg/rullama_bg.wasm` + TypeScript typings into
the app's `pkg/`. (Engine internals — kernels, GGUF, towers, parity — are
documented in the rullama-framework engine repo.)

### The PWA

The user-facing browser app lives in `apps/web/` — a production-quality React + Vite
+ Tailwind + Workbox chat PWA (service-worker offline shell, restart dialog on
deploys, attachment UI, conversation history in OPFS + SQLite via
`rsqlite-wasm`) built against the shared wasm bundle.

```sh
# React / Vite PWA — auto-runs the wasm bundle build via `pnpm dev`.
cd apps/web
pnpm install
pnpm dev                 # https://localhost:5173/
```

The first load streams the ~7 GB blob from the local Ollama install (or an R2
mirror — see `scripts/upload-models-to-r2.sh`) through a Dedicated Worker that
owns a `FileSystemSyncAccessHandle` over OPFS. Bytes go network → sync handle
→ disk without ever pinning a Blob in the JS heap. Subsequent loads (within
the same Safari session) reuse the cached file.

### iPhone scripted runs

The PWA is fully drivable from the Mac via Apple's `safaridriver`:

```sh
# One-time setup on the phone:
#   Settings → Safari → Advanced → Remote Automation = on
#                                  Web Inspector       = on
#                                  Feature Flags → WebGPU = on
# Then on the Mac:
safaridriver -p 4444 &
./apps/web/serve-iphone.sh            # HTTPS serve reachable from the phone's Safari
./apps/web/test/iphone-test.sh        # navigate → Load → chat → log perf
```

`/tmp/rullama-page.log` collects beacon traces from the page (`[chat]`,
`[pe]`, `[tg]`, `[gen]`, `[wkr]`, `[rs]`) so any regression in a phone
run leaves a server-side trail even after a WebContent crash.

## `apps/native/` — the desktop + mobile app

A **.NET 9 / Avalonia 11** app (`apps/native/app/`, `Rullama.sln`) over a Rust
**C-ABI shim** (`apps/native/rust-core/`, a `cdylib`/`staticlib`) that links the engine
crates directly — no HTTP, no wasm. Runs on macOS / Windows / Linux desktop
(builds with the plain `dotnet` SDK, no Xcode) plus an Android head; iOS needs
Xcode 16. The engine `Model` is `!Send`, so each handle owns one OS thread and
calls are marshalled to it inside Rust (`Avalonia → RustCore P/Invoke → rust-core
→ rullama-engine` on wgpu Metal/DX12/Vulkan). Chat, multimodal, tools, voice
(Kokoro TTS), in-app model downloads, LoRA fine-tuning, RAG, voice-cloning, and
ROME editing are wired. Shipped through the app stores.

```bash
cd apps/native
cargo build --manifest-path rust-core/Cargo.toml --release   # build the C-ABI core
./scripts/build-rust.sh release                              # stage the native lib
dotnet run --project app/Rullama.ProbeSmoke                  # verify the P/Invoke path
dotnet run --project app/Rullama.Desktop                     # run the desktop app
```

`rust-core` is excluded from the root workspace (it links wgpu); it path-deps
`../../../../rullama-framework/engine/{rullama-engine,rullama-lora}`. See
[`apps/native/README.md`](apps/native/README.md) for the full toolchain + heads.

## `apps/cli/` — the agentic CLI

An AI agentic coding CLI (installed binary: **`rullama`**) — multi-agent
orchestrator/worker decomposition, a rich tool system (file/bash/git/web), an MCP
client, interactive / single-shot / batch / TUI modes, plan + task management, and
semantic conversation memory. **BYOK**: on first run it prompts you to pick a
provider (Ollama-local / OpenAI / Anthropic / …) — no hosted account.

```bash
cd apps/cli
cargo build --release
cargo install --path .        # installs the `rullama` binary
rullama                       # first run: pick + configure a provider
```

`apps/cli/` is its own Cargo workspace (own `Cargo.lock`), excluded from the root and
path-depping the `../../../rullama-framework` harness crates. The legacy Studio
remote-bridge (web control of CLI agents) is kept behind the off-by-default
`remote-bridge` feature. See [`apps/cli/README.md`](apps/cli/README.md).

## Docker / deploy

`compose.yaml` packages the built PWA + a model-blob HTTP service behind
nginx, designed to sit behind Cloudflare. The Cargo workspace ships
`cargo docker:*` aliases (dispatched through the `xtask` crate) so the
deploy loop doesn't need shell aliases:

| Alias                  | Effective command                                                              |
|------------------------|--------------------------------------------------------------------------------|
| `cargo docker:build`   | `docker compose build`                                                         |
| `cargo docker:start`   | `docker compose up -d`                                                         |
| `cargo docker:stop`    | `docker compose down`                                                          |
| `cargo docker:restart` | `docker compose build --no-cache` then `docker compose up -d --force-recreate` |
| `cargo docker:logs`    | `docker compose logs -f --tail=200`                                            |
| `cargo docker:ps`      | `docker compose ps`                                                            |

First run compiles `xtask` (~1 s); subsequent invocations reuse the cached
binary. Add new tasks by appending a match arm in `xtask/src/main.rs` and
a corresponding line in `.cargo/config.toml`. The compose file's
`OLLAMA_MODELS_DIR` env var picks the host's model store; defaults to
`/usr/share/ollama/.ollama/models`.

## Engine parity checks (run from rullama-framework)

The engine's native parity examples (`greedy_parity`, `model_api`,
`chained_smoke`, …) run the same code paths against host wgpu (Metal on macOS,
Vulkan on Linux) without a browser. **They live with the engine**, not in this
repo — run them from the sibling
[`rullama-framework`](https://github.com/Brainwires/rullama-framework) engine
checkout against `rullama-engine`, e.g.:

```sh
cd ../rullama-framework/engine
cargo run -p rullama-engine --release --example greedy_parity -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" 5
```

See the engine repo's README for the full list.

## Fine-tuning

`rullama-lora` runs LoRA SGD against the live wgpu kernels — no Burn, no
PyTorch, no separate runtime. Scope: rank-r LoRAs on
`attn_q` / `attn_k` / `attn_v` / `attn_o` and the FFN projections, Adam, global
L2 grad clipping, gradient accumulation, mixed precision, gradient
checkpointing. PerPosition CE is a single-forward variant with a ~C/2 speedup
vs. the multi-forward path.

**In the browser:** the unified wasm bundle (see [Get the wasm bundle](#get-the-wasm-bundle-from-the-engine))
exposes `TrainingSession` to JS alongside `Model`. The Fine-tune tab in
`apps/web/` drives a full session — dataset upload, hyperparam UI, live
loss chart, save adapter to OPFS as safetensors. The same `Model` that's
loaded for inference accepts the trained adapter via `Model.loadAdapter(bytes)`
(re-runs in the chat tab against the adapted weights).

**Native:** the native trainer examples (`overfit_one`, `train_jsonl`,
`eval_adapter`) live with the engine now — run them from the rullama-framework engine
repo against `rullama-lora` (e.g.
`cargo run -p rullama-lora --release --example overfit_one -- <gguf>`). See
the engine repo's README.

## Web app architecture (`apps/web/`)

```
PWA (host page) ──┐
                  ▼  postMessage RPC
  ┌──────────────────────────────────────────────────────────────────┐
  │ inference-worker.js (Dedicated Worker)                          │
  │   ▶ owns FileSystemSyncAccessHandle for the GGUF                │
  │   ▶ owns the wasm Model handle                                  │
  │     ┌──────────────────────────────────────────────────────┐    │
  │     │ wasm32 (Rust, the rullama-engine bundle)             │    │
  │     │   Model.loadFromOpfs(read_fn, total)                 │    │
  │     │           │                                          │    │
  │     │           ▼                                          │    │
  │     │   GgufReader (header only, ~5 MB)                    │    │
  │     │           │                                          │    │
  │     │           │ TensorFetcher (OPFS sync read | HTTP Range)│
  │     │           ▼                                          │    │
  │     │   WeightCache  ─────────▶  Forward / VisionForward / │    │
  │     │   (lazy GPU upload,         GpuAudioForward          │    │
  │     │    per-tile range fetch     (per-layer encoder       │    │
  │     │    on big tensors)           submits, GPU-resident   │    │
  │     │                              KV cache)               │    │
  │     │                                  │                   │    │
  │     │                                  ▼                   │    │
  │     │                      wgpu (WebGPU / Metal / Vulkan)  │    │
  │     │                                  │                   │    │
  │     │                                  ▼                   │    │
  │     │      WGSL kernels: matmul Q4_K/Q6_K/F16, rmsnorm,    │    │
  │     │      rmsnorm_per_row, rope_neox, attention (incl.    │    │
  │     │      HPD-f16 + block-local + subgroup variants),     │    │
  │     │      conv2d, geglu, softcap, residual_add, scale,    │    │
  │     │      top_k, quick_gelu, plus backward kernels for    │    │
  │     │      training (cross_entropy, rmsnorm, rope, geglu,  │    │
  │     │      attention dQ / dKV, matmul Q4_K / Q6_K, Adam)   │    │
  │     └──────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────────────────────┘
                  │
                  ▲  postMessage replies (tokens, errors)
PWA renders tokens, manages chat history, handles attachments.
```

The Worker move (M15) is what unblocked iPhone inference: iOS Safari only
exposes `FileSystemSyncAccessHandle` in Worker contexts, and the Worker
isolates inference from main-thread page-watchdog reapers.

The reference Go implementation lives in Ollama's tree under
`model/models/gemma4/`. Every op in the engine's `reference/forward.rs`
(CPU oracle), `forward_chained.rs` (production GPU forward),
`multimodal/vision.rs`, and `multimodal/audio.rs` corresponds 1:1 — see the
`rullama-engine` crate in the [`rullama-framework`](../rullama-framework) repo.

## Web app performance (`apps/web/`)

Measurements as of M15:

| Target                   | Steady-state tok/s (gen) | Notes                                  |
|--------------------------|--------------------------|----------------------------------------|
| iPhone 16e (A18, iOS 26) | **~4.65 tok/s**          | text-only, `max_context=512`           |
| AMD Radeon Pro 555 (Mac) | ~1 tok/s (M7 baseline)   | naive kernels, tiled matmul deferred   |

The architectural foundation (chained encoder, GPU-resident KV cache, per-layer
submits, per-tile range fetch from OPFS) is in place. Inference kernels are
still naive matvec; reaching ≥10 tok/s on both Mac and phone needs tiled
matmul + bind-group caching + kernel fusion (the M8 line on the roadmap).

The iPhone A18 advertises 1 GiB for both `max_buffer_size` and
`max_storage_buffer_binding_size` — four times the WebGPU spec floor — so
there's real headroom for fewer/larger weight buffers (currently 455 of
them resident, see M15 follow-ups).

Other capability notes captured during iPhone validation:
- `shader-f16` ✓ — packed FP16 MADs engage on A18.
- `timestamp-query` ✓ — Pro 555 doesn't expose this; could wire GPU-side
  per-pass timing.
- `subgroups` ✗ — A18 has SIMDgroup hardware but Safari's WebGPU doesn't
  surface WGSL subgroup ops yet. Vision attention falls through to the
  no-subgroup HPD-f16 kernel automatically.

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

Contributions are accepted under the same dual-license terms.
