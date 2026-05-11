# rullama

Browser-resident **Gemma 4 inference** in pure Rust → WebAssembly + WebGPU.
Loads the same GGUF blobs Ollama already has on disk, runs the forward pass on
your local GPU through hand-written WGSL, never touches a remote server.

The intent is a **PWA-pluggable inference engine**, not a port of Ollama-the-server.
Ollama has 275K LOC of Go that wraps llama.cpp via CGO plus model registry, CLI,
conversion tooling, multimodal pipelines — almost none of which apply to a
browser library. What survives the scope cut is the *core inference path* over
Ollama's storage format.

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
- ❌ MoE `gemma4:26b` / `gemma4:31b` — out of scope.
- ❌ Other architectures (llama, mistral, qwen, phi).
- 🛠️ **Mobile multimodal** — desktop multimodal works; the iPhone loader
  currently skips the vision/audio towers to fit in shared RAM. Lazy
  upload for those is a follow-up.

## Quickstart

You need:
- Rust ≥ 1.91 + `wasm-pack`
- A WebGPU-capable browser (Chrome 113+, Edge 113+, recent Firefox; iOS
  Safari 17.4+ for phones)
- Ollama installed locally with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`)

```sh
# Build the wasm bundle (writes /pkg/rullama.js + rullama_bg.wasm)
wasm-pack build --target web --release

# Start the dev HTTPS server (scans ~/.ollama/models, serves blobs with Range
# support, sets the COOP/COEP headers WebGPU + cross-origin-isolated need).
# Self-signed cert at ~/.local/share/rullama/{cert,key}.pem.
CERT_FILE=~/.local/share/rullama/cert.pem KEY_FILE=~/.local/share/rullama/key.pem \
    ./examples/pwa/serve.sh

# Desktop:  https://localhost:8088/examples/pwa/index.html
# iPhone:   https://<mac-lan-ip>:8088/examples/pwa/index.html
# Pick gemma4:e2b → Load → chat.
```

The first load streams the ~7 GB blob from the local Ollama install through
a Dedicated Worker that owns a `FileSystemSyncAccessHandle` over OPFS — bytes
go network → sync handle → disk without ever pinning a Blob in the JS heap.
Subsequent loads (within the same Safari session) reuse the cached file.

### iPhone scripted runs

The PWA is fully drivable from the Mac via Apple's `safaridriver`:

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

## Native sanity checks

The same code paths run natively against host wgpu (Metal on macOS, Vulkan on
Linux). Useful for parity testing without a browser:

```sh
# Greedy parity vs Ollama (CPU oracle)
cargo run --release --features cpu-reference --example greedy_parity -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" 5

# Full-stack chat through the public Model API
cargo run --release --features cpu-reference --example model_api -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --greedy --max=16

# Standalone chained forward (M7 perf path)
cargo run --release --features cpu-reference --example chained_smoke -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --max=8
```

`--features cpu-reference` enables the f32 oracle path for parity testing; the
production path is always built.

## Architecture

```
PWA (host page) ──┐
                  ▼  postMessage RPC
  ┌──────────────────────────────────────────────────────────────────┐
  │ inference-worker.js (Dedicated Worker)                          │
  │   ▶ owns FileSystemSyncAccessHandle for the GGUF                │
  │   ▶ owns the wasm Model handle                                  │
  │     ┌──────────────────────────────────────────────────────┐    │
  │     │ wasm32 (Rust, our crate)                             │    │
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
  │     │      HPD-f16 + block-local), conv2d, geglu,          │    │
  │     │      softcap, residual_add, scale, top_k, quick_gelu │    │
  │     └──────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────────────────────┘
                  │
                  ▲  postMessage replies (tokens, errors)
PWA renders tokens, manages chat history, handles attachments.
```

The Worker move (M15) is what unblocked iPhone inference: iOS Safari only
exposes `FileSystemSyncAccessHandle` in Worker contexts, and the Worker
isolates inference from main-thread page-watchdog reapers.

Reference Go implementation:
`/Users/nightness/Source/ollama/model/models/gemma4/`. Every op in
`src/reference/forward.rs` (CPU oracle), `src/reference/forward_chained.rs`
(production GPU forward), `src/multimodal/vision.rs`, and
`src/multimodal/audio.rs` corresponds 1:1.

## Performance

Measurements as of M15:

| Target                   | Steady-state tok/s (gen) | Notes                                  |
|--------------------------|--------------------------|----------------------------------------|
| iPhone 16e (A18, iOS 26) | **~4.65 tok/s**          | text-only, `max_context=512`           |
| AMD Radeon Pro 555 (Mac) | ~1 tok/s (M7 baseline)   | naive kernels, tiled matmul deferred   |

The architectural foundation (chained encoder, GPU-resident KV cache, per-layer
submits, per-tile range fetch from OPFS) is in place. Kernels are still naive
matvec; reaching the ≥10 tok/s target on both Mac and phone needs tiled
matmul + bind-group caching + kernel fusion. See the M8 milestones in
`/Users/nightness/.claude/plans/greenfield-port-of-source-ollama-velvet-treehouse.md`.

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
src/
├── api.rs                    # JS-facing Model: load / loadFromUrl / loadFromOpfs[TextOnly]
├── backend/
│   ├── context.rs            # WgpuCtx (device, queue, adapter limits)
│   ├── dispatch.rs           # cached + chained kernel dispatchers
│   ├── pipelines.rs          # one ComputePipeline per kernel (built once)
│   ├── weight_cache.rs       # lazy GPU upload, per-tile range fetch on big tensors
│   ├── matmul.rs / elementwise.rs / spike.rs    # one-shot dispatchers (parity tests)
├── gguf/
│   ├── reader.rs             # GGUF v3 parser (header + tensor descriptors)
│   ├── fetcher.rs            # TensorFetcher trait + InMemoryFetcher + HttpRangeFetcher + OpfsFetcher
│   ├── tensor.rs             # dequant_tensor_to_f32 / dequant_row_to_f32 (sync + async)
│   ├── quant.rs / dtype.rs / value.rs
├── kernels/wgsl/             # 25+ hand-written compute shaders (text + vision + audio)
├── model/config.rs           # Gemma4Config: parses gemma4.* metadata keys
├── multimodal/
│   ├── vision.rs             # ViT forward (16 blocks, 768d, ClippableLinear)
│   ├── audio.rs              # Conformer forward (12 blocks, 1024d, block-local attention)
│   └── audio_features.rs     # WAV → 128-bin log-mel (realfft)
├── reference/
│   ├── forward.rs            # CPU f32 forward (parity oracle)
│   ├── forward_gpu.rs        # M3-era GPU forward with per-kernel readbacks (oracle)
│   ├── forward_chained.rs    # M7 production GPU forward, per-layer submits (M15)
│   ├── ops.rs / weights.rs
├── sampling.rs               # temperature, top-k, top-p, rep penalty
├── template/gemma4_small.rs  # chat-template renderer (matches Ollama)
└── tokenizer/                # GGUF BPE tokenizer (Ollama-bit-exact)

examples/
├── pwa/
│   ├── index.html                  # the demo PWA
│   ├── inference-worker.js         # M15 Dedicated Worker, owns Model + sync OPFS handle
│   ├── opfs-store.js               # OPFS download + read API (main-thread)
│   ├── opfs-writer-worker.js       # streams GGUF → OPFS via SyncAccessHandle.write
│   ├── serve.sh                    # dev HTTPS server + /api/log /api/blob endpoints
│   ├── run-on-iphone.sh            # navigate → Load → chat over safaridriver
│   ├── iphone-session-keeper.sh    # long-lived session so OPFS persists across runs
│   ├── clean-iphone.sh             # wipe OPFS + IDB rullama-models
│   ├── bench-on-iphone.sh          # kernel-level WebGPU bench harness
│   ├── chat.js / image_preprocess.js / cache.js
│   └── bench.html
├── greedy_parity.rs          # CPU forward greedy vs Ollama
├── chained_smoke.rs          # standalone Forward driver
├── model_api.rs              # public Model API end-to-end
├── vision_parity.rs          # vision tower vs Ollama (M11)
├── audio_parity.rs           # audio tower vs Ollama (M13)
└── inspect.rs / decode_ids.rs / encode_check.rs / forward_smoke.rs / list_tensors.rs
```

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

Contributions are accepted under the same dual-license terms.
