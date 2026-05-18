# rullama

Browser-resident **Gemma 4 inference** in pure Rust → WebAssembly + WebGPU.
Loads the same GGUF blobs Ollama already has on disk, runs the forward pass on
your local GPU through hand-written WGSL, never touches a remote server.

The intent is a **PWA-pluggable inference engine**, not a port of Ollama-the-server.
Ollama has 275K LOC of Go that wraps llama.cpp via CGO plus model registry, CLI,
conversion tooling, multimodal pipelines — almost none of which apply to a
browser library. What survives the scope cut is the *core inference path* over
Ollama's storage format.

## Workspace

A two-crate Cargo workspace:

| Crate              | Path                       | Target       | Status   |
|--------------------|----------------------------|--------------|----------|
| `rullama`          | `crates/rullama`           | wasm + native | release-track |
| `rullama-finetune` | `crates/rullama-finetune`  | wasm + native | LoRA SGD over the same wgpu kernels; PWA exposes `TrainingSession` in the Fine-tune tab |

The iOS bench harness (`tools/ios-bench`) is a sibling crate excluded from the
workspace so `cargo build --workspace --target wasm32-unknown-unknown` doesn't
try to compile its staticlib for wasm.

## What works today

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
- ✅ **In-browser LoRA fine-tuning (`rullama-finetune`, wasm + native).**
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

## Quickstart

You need:
- Rust ≥ 1.91 + `wasm-pack` (`cargo install wasm-pack --locked --version 0.13.1`)
- A WebGPU-capable browser (Chrome 113+, Edge 113+, recent Firefox; iOS
  Safari 17.4+ for phones)
- Ollama installed locally with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`)

### Build the wasm bundle

```sh
# Unified bundle — exposes both inference (`Model`) and training
# (`TrainingSession`) wasm-bindgen surfaces. Built from `rullama-finetune`
# because that's the crate that re-exports both. `--out-name rullama` keeps
# the JS entry at `pkg/rullama.js` for PWA import compatibility.
wasm-pack build crates/rullama-finetune --target web --release \
    --out-dir ../../pkg --out-name rullama

# Inference-only variant (smaller bundle, no TrainingSession). Use when
# shipping a chat-only deployment.
wasm-pack build crates/rullama --target web --release --out-dir ../../pkg
```

This emits `pkg/rullama.js` + `pkg/rullama_bg.wasm` + TypeScript typings.

### Two example PWAs

The repo ships two browser harnesses against the same wasm bundle:

| Path             | Stack                | Purpose                                                         |
|------------------|----------------------|-----------------------------------------------------------------|
| `examples/web/`  | React + Vite + Tailwind + Workbox SW | Production-quality chat PWA — service-worker-based offline shell, restart dialog on deploys, attachment UI, conversation history in OPFS + SQLite (`rsqlite-wasm`). |
| `examples/pwa/`  | Vanilla JS + bash    | Bench harness and `safaridriver`-driven scripted iPhone runs.   |

Pick `examples/web/` for hacking on the user-facing chat experience; pick
`examples/pwa/` for kernel benchmarks or hands-off iPhone perf reruns.

```sh
# React / Vite PWA — auto-runs the wasm bundle build via `pnpm dev`.
cd examples/web
pnpm install
pnpm dev                 # https://localhost:5173/

# Vanilla bench / iPhone harness — needs the wasm bundle built first.
CERT_FILE=~/.local/share/rullama/cert.pem KEY_FILE=~/.local/share/rullama/key.pem \
    ./examples/pwa/serve.sh
# Desktop:  https://localhost:8088/examples/pwa/index.html
# iPhone:   https://<mac-lan-ip>:8088/examples/pwa/index.html
```

The first load streams the ~7 GB blob from the local Ollama install (or an R2
mirror — see `scripts/upload-models-to-r2.sh`) through a Dedicated Worker that
owns a `FileSystemSyncAccessHandle` over OPFS. Bytes go network → sync handle
→ disk without ever pinning a Blob in the JS heap. Subsequent loads (within
the same Safari session) reuse the cached file.

### iPhone scripted runs

The vanilla PWA is fully drivable from the Mac via Apple's `safaridriver`:

```sh
# One-time setup on the phone:
#   Settings → Safari → Advanced → Remote Automation = on
#                                  Web Inspector       = on
#                                  Feature Flags → WebGPU = on
# Then on the Mac:
safaridriver -p 4444 &
./examples/pwa/iphone-session-keeper.sh &        # keep an OPFS scope alive
./examples/pwa/run-on-iphone.sh                  # navigate → Load → chat → log perf
./examples/pwa/clean-iphone.sh                   # wipe OPFS between trials
```

`/tmp/rullama-page.log` collects beacon traces from the page (`[chat]`,
`[pe]`, `[tg]`, `[gen]`, `[wkr]`, `[rs]`) so any regression in a phone
run leaves a server-side trail even after a WebContent crash.

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

## Native sanity checks

The same code paths run natively against host wgpu (Metal on macOS, Vulkan on
Linux). Useful for parity testing without a browser:

```sh
# Greedy parity vs Ollama (CPU oracle)
cargo run -p rullama --release --features cpu-reference --example greedy_parity -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" 5

# Full-stack chat through the public Model API
cargo run -p rullama --release --features cpu-reference --example model_api -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --greedy --max=16

# Standalone chained forward (M7 perf path)
cargo run -p rullama --release --features cpu-reference --example chained_smoke -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --max=8
```

`--features cpu-reference` is now a no-op (the f32 oracle is always built); the
flag is kept so existing scripts keep working.

## Fine-tuning

`rullama-finetune` runs LoRA SGD against the live wgpu kernels — no Burn, no
PyTorch, no separate runtime. Scope: rank-r LoRAs on
`attn_q` / `attn_k` / `attn_v` / `attn_o` and the FFN projections, Adam, global
L2 grad clipping, gradient accumulation, mixed precision, gradient
checkpointing. PerPosition CE is a single-forward variant with a ~C/2 speedup
vs. the multi-forward path.

**In the browser:** the unified wasm bundle (see [Build](#build-the-wasm-bundle))
exposes `TrainingSession` to JS alongside `Model`. The Fine-tune tab in
`examples/web/` drives a full session — dataset upload, hyperparam UI, live
loss chart, save adapter to OPFS as safetensors. The same `Model` that's
loaded for inference accepts the trained adapter via `Model.loadAdapter(bytes)`
(re-runs in the chat tab against the adapted weights).

**Native:**

```sh
# Overfit a single (prompt, target) pair — acceptance test that the
# backward path and Adam are wired correctly.
cargo run -p rullama-finetune --release --example overfit_one -- \
    ~/.ollama/models/blobs/sha256-<digest>

# Train on a JSONL dataset. See `crates/rullama-finetune/examples/data/echo.jsonl`
# for the format; env knobs documented in the example's docstring.
cargo run -p rullama-finetune --release --example train_jsonl -- \
    ~/.ollama/models/blobs/sha256-<digest> \
    crates/rullama-finetune/examples/data/echo.jsonl

# End-to-end smoke: train an adapter, save safetensors, reload via the
# public Model API, run a generation against the adapted weights.
cargo run -p rullama-finetune --release --example eval_adapter -- \
    ~/.ollama/models/blobs/sha256-<digest> /path/to/adapter.safetensors
```

## Architecture

```
PWA (host page) ──┐
                  ▼  postMessage RPC
  ┌──────────────────────────────────────────────────────────────────┐
  │ inference-worker.js (Dedicated Worker)                          │
  │   ▶ owns FileSystemSyncAccessHandle for the GGUF                │
  │   ▶ owns the wasm Model handle                                  │
  │     ┌──────────────────────────────────────────────────────┐    │
  │     │ wasm32 (Rust, the rullama crate)                     │    │
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
`model/models/gemma4/`. Every op in `crates/rullama/src/reference/forward.rs`
(CPU oracle), `forward_chained.rs` (production GPU forward),
`multimodal/vision.rs`, and `multimodal/audio.rs` corresponds 1:1.

## Performance

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

## Layout

```
crates/rullama/
├── src/
│   ├── api.rs                    # JS-facing Model: load / loadFromUrl / loadFromOpfs[TextOnly] / loadAdapter / clearAdapter
│   ├── lora.rs                   # InferenceAdapter — parses the safetensors blob TrainingSession writes
│   ├── backend/
│   │   ├── context.rs            # WgpuCtx (device, queue, adapter limits)
│   │   ├── dispatch.rs           # cached + chained kernel dispatchers (incl. backward + Adam)
│   │   ├── pipelines.rs          # one ComputePipeline per kernel (built once)
│   │   ├── weight_cache.rs       # lazy GPU upload, per-tile range fetch on big tensors
│   │   ├── matmul.rs / elementwise.rs / spike.rs    # one-shot dispatchers (parity tests)
│   ├── gguf/
│   │   ├── reader.rs             # GGUF v3 parser (header + tensor descriptors)
│   │   ├── fetcher.rs            # TensorFetcher trait + In-memory / HttpRange / Opfs impls
│   │   ├── tensor.rs             # dequant_tensor_to_f32 / dequant_row_to_f32 (sync + async)
│   │   ├── quant.rs / dtype.rs / value.rs
│   ├── kernels/wgsl/             # 70+ hand-written compute shaders (text + vision + audio + backward)
│   ├── model/config.rs           # Gemma4Config: parses gemma4.* metadata keys
│   ├── multimodal/
│   │   ├── vision.rs             # ViT forward (16 blocks, 768d, ClippableLinear)
│   │   ├── audio.rs              # Conformer forward (12 blocks, 1024d, block-local attention)
│   │   └── audio_features.rs     # WAV → 128-bin log-mel (realfft)
│   ├── reference/
│   │   ├── forward.rs            # CPU f32 forward (parity oracle)
│   │   ├── forward_gpu.rs        # M3-era GPU forward with per-kernel readbacks (oracle)
│   │   ├── forward_chained.rs    # M7 production GPU forward, per-layer submits (M15)
│   │   ├── ops.rs / weights.rs
│   ├── sampling.rs               # temperature, top-k, top-p, rep penalty
│   ├── template/gemma4_small.rs  # chat-template renderer (matches Ollama)
│   └── tokenizer/                # GGUF BPE tokenizer (Ollama-bit-exact)
└── examples/
    ├── greedy_parity.rs          # CPU forward greedy vs Ollama
    ├── chained_smoke.rs          # standalone Forward driver
    ├── model_api.rs              # public Model API end-to-end
    ├── vision_parity.rs          # vision tower vs Ollama (M11)
    ├── audio_parity.rs           # audio tower vs Ollama (M13)
    ├── matmul_bench.rs           # native wgpu matmul microbench
    └── inspect.rs / decode_ids.rs / encode_check.rs / list_tensors.rs / …

crates/rullama-finetune/
├── src/
│   ├── shared/                   # vendored config / error / progress types
│   ├── dataset_loader.rs         # JSONL parser + Tokenizer trait
│   ├── lr_schedule.rs            # warmup + linear / cosine / cosine-warm-restarts
│   ├── lora.rs                   # per-LoRA GPU state (A / B), grad buffers
│   ├── scratch.rs                # per-step GPU scratch buffers for backward
│   ├── wasm_bindgen_api.rs       # JS-facing TrainingSession (wasm32 only)
│   └── session.rs                # TrainingSession — forward → loss → backward → Adam
└── examples/
    ├── overfit_one.rs            # single-pair acceptance test
    ├── train_jsonl.rs            # JSONL dataset trainer
    ├── eval_adapter.rs           # load a trained safetensors blob and generate
    └── data/echo.jsonl

examples/
├── web/                          # React + Vite + Tailwind + Workbox SW production demo
│   └── src/components/FineTunePanel.tsx  # in-browser LoRA training tab over the loaded Model
└── pwa/                          # Vanilla JS bench harness + safaridriver scripts
    ├── index.html / bench.html
    ├── inference-worker.js       # Dedicated Worker — owns Model + sync OPFS handle
    ├── opfs-store.js             # OPFS download + read API (main-thread)
    ├── opfs-writer-worker.js     # streams GGUF → OPFS via SyncAccessHandle.write
    ├── serve.sh                  # dev HTTPS server + /api/log /api/blob endpoints
    ├── run-on-iphone.sh / iphone-session-keeper.sh / clean-iphone.sh
    └── bench-on-iphone.sh

tools/ios-bench/                  # staticlib for Xcode — C-ABI rullama_run_bench
docker/                           # nginx + R2 mirror configs
scripts/                          # ops scripts (model upload, etc.)
```

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

Contributions are accepted under the same dual-license terms.
