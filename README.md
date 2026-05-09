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

- ✅ **`gemma4:e2b`** loads end-to-end and generates greedy output bit-identical
  to Ollama. (`gemma4:e4b` is shape-compatible — pull and try it.)
- ✅ **Q4_K + Q6_K + F16 + F32** quants (the actual mix in `gemma4:e2b` Q4_K_M).
- ✅ **Streaming load** via HTTP byte-range requests — the 7 GB GGUF never
  enters wasm linear memory in bulk, dodging the 4 GB cap.
- ✅ **IndexedDB blob cache** — second page load is instant, no re-download.
- ✅ **Multi-turn chat** with system prompt, mid-generation Stop, persistent
  KV cache.
- ✅ **One CommandEncoder per token** GPU forward (M7).
- ❌ MoE `gemma4:26b` / `gemma4:31b` — out of scope.
- ❌ Other architectures (llama, mistral, qwen, phi).
- ❌ Vision / audio multimodal towers.

## Quickstart

You need:
- Rust ≥ 1.91 + `wasm-pack`
- A WebGPU-capable browser (Chrome 113+, Edge 113+, recent Firefox/Safari)
- Ollama installed locally with `gemma4:e2b` pulled (`ollama pull gemma4:e2b`)

```sh
# Build the wasm bundle
wasm-pack build --target web --release

# Start the dev server (scans ~/.ollama/models, serves blobs with Range support)
./examples/pwa/serve.sh

# Open http://localhost:8088/examples/pwa/index.html
# Pick gemma4:e2b → Load → chat.
```

The first load streams the ~7 GB blob from the local Ollama install into
IndexedDB (a few seconds on a fast disk). Subsequent loads come straight from
the cache and skip the network entirely.

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
PWA (host page)                ──▶ Model.loadFromUrl(url)
                                       │
                                       ▼
JS shim ──▶ wasm32 (Rust)              │
                                       ▼
GgufReader (header only, ~5 MB)        │   ┌─ scratch buffers
        │                              │   │  hidden, q, k, v, ffn_*, ple_*
        │ TensorFetcher (HTTP Range)   │   │  per-layer KV cache
        ▼                              ▼   │
WeightCache ────────────────▶ Forward struct
        (lazy GPU upload,             │  one CommandEncoder per token,
         bytes dropped after)         │  one queue.submit, one logits readback
                                      ▼
                              wgpu (WebGPU / Metal / Vulkan)
                                      │
                                      ▼
                          WGSL kernels (matmul Q4_K/Q6_K/F16, rmsnorm,
                          rmsnorm_per_row, rope_neox, attention, geglu,
                          softcap, residual_add, scale)
```

Reference Go implementation:
`/Users/nightness/Source/ollama/model/models/gemma4/model_text.go`. Every
op in `src/reference/forward.rs` (CPU oracle) and
`src/reference/forward_chained.rs` (production GPU forward) corresponds 1:1.

## Performance

On Apple M-series, gemma4:e2b runs at roughly 1 tok/s as of M7. Reaching the
≥10 tok/s target is open work — the architectural foundation (chained
encoder, GPU-resident KV cache, single per-token readback) is in place but
the kernels themselves are still naive matvec. See the M8 milestones in
`/Users/nightness/.claude/plans/greenfield-port-of-source-ollama-velvet-treehouse.md`
for the perf hardening plan (tiled matmul, bind-group caching, kernel fusion).

The M6 streaming load lifts the wasm32 4 GB cap, so the 7 GB e2b GGUF fits in
the browser regardless of perf.

## Layout

```
src/
├── api.rs                    # JS-facing Model + ChatMessage + GenerateOptions
├── backend/
│   ├── context.rs            # WgpuCtx (device, queue, adapter)
│   ├── dispatch.rs           # cached + chained kernel dispatchers
│   ├── pipelines.rs          # one ComputePipeline per kernel (built once)
│   ├── weight_cache.rs       # lazy GPU upload of GGUF tensors
│   ├── matmul.rs / elementwise.rs / spike.rs    # one-shot dispatchers (parity tests)
├── gguf/
│   ├── reader.rs             # GGUF v3 parser (header + tensor descriptors)
│   ├── fetcher.rs            # TensorFetcher trait + InMemoryFetcher + HttpRangeFetcher
│   ├── tensor.rs             # dequant_tensor_to_f32 / dequant_row_to_f32 (sync + async)
│   ├── quant.rs / dtype.rs / value.rs
├── kernels/wgsl/             # 12 hand-written compute shaders
├── model/config.rs           # Gemma4Config: parses gemma4.* metadata keys
├── reference/
│   ├── forward.rs            # CPU f32 forward (parity oracle)
│   ├── forward_gpu.rs        # M3-era GPU forward with per-kernel readbacks (oracle)
│   ├── forward_chained.rs    # M7 production GPU forward (one encoder/token)
│   ├── ops.rs / weights.rs
├── sampling.rs               # temperature, top-k, top-p, rep penalty
├── template/gemma4_small.rs  # chat-template renderer (matches Ollama)
└── tokenizer/                # GGUF BPE tokenizer (Ollama-bit-exact)

examples/
├── pwa/                      # the demo PWA + dev server
├── greedy_parity.rs          # CPU forward greedy vs Ollama
├── chained_smoke.rs          # standalone Forward driver
├── model_api.rs              # public Model API end-to-end
└── inspect.rs / decode_ids.rs / encode_check.rs / forward_smoke.rs / list_tensors.rs
```

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT License ([LICENSE-MIT](./LICENSE-MIT))

at your option.

Contributions are accepted under the same dual-license terms.
