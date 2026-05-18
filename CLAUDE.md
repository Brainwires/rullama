# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Browser-resident Gemma 4 inference in pure Rust → WebAssembly + WebGPU. Loads
Ollama's on-disk GGUF blobs (no server) and runs the forward pass on the local
GPU through hand-written WGSL. **Scope is intentionally narrow**: Gemma 4 only
(`gemma4:e2b`, `gemma4:e4b`), `Q4_K_M` mix only (`Q4_K` / `Q6_K` / `F16` / `F32`).
Other architectures and MoE variants are out of scope.

The reference Go impl lives in Ollama's tree at `model/models/gemma4/`. Ops in
`crates/rullama/src/reference/forward.rs` (CPU oracle) and `forward_chained.rs`
(production GPU forward) correspond **1:1** to it — when adding or changing an
op, diff against Ollama, not a llama.cpp port.

## Workspace layout

Two-crate Cargo workspace + a sibling iOS bench crate that is intentionally
**excluded** from the workspace (so `cargo build --workspace --target
wasm32-unknown-unknown` doesn't try to compile its staticlib for wasm):

| Crate              | Target        | Notes                                             |
|--------------------|---------------|---------------------------------------------------|
| `crates/rullama`           | wasm + native | The engine. Stable public API is `api` + `error` + `sampling` + `lora` ONLY; everything else (`backend`, `gguf`, `kernels`, `model`, `multimodal`, `reference`, `template`, `tokenizer`) is `#[doc(hidden)]` and may change in any patch release. |
| `crates/rullama-finetune`  | wasm + native | LoRA SGD over the same wgpu kernels. On wasm32 it exposes a `TrainingSession` wasm-bindgen surface; the PWA's Fine-tune tab consumes it. Native examples (`overfit_one`, `train_jsonl`, `eval_adapter`) are the parity oracle. |
| `xtask`                    | native        | Tiny std-only dispatcher for `cargo docker:*` aliases. Keep it dependency-free. |
| `tools/ios-bench`          | static lib    | Out-of-workspace; staticlib for Xcode, C-ABI `rullama_run_bench`. |

Rust toolchain is **pinned to 1.91** via `rust-toolchain.toml`; `wasm32-unknown-unknown`
is a required target. Workspace deps in root `Cargo.toml` are pinned to recent
minors (e.g. `wgpu = "29.0.3"`), not bare majors — match that style when adding
deps.

## Build / run

```sh
# WASM bundle — output lands at <repo>/pkg/ and is shared by BOTH example PWAs
# Build via the finetune crate so the unified bundle exposes both inference
# (Model) and training (TrainingSession) wasm-bindgen surfaces; --out-name
# keeps the JS entry at pkg/rullama.js for PWA import compat.
wasm-pack build crates/rullama-finetune --target web --release --out-dir ../../pkg --out-name rullama

# Inference-only variant (smaller bundle, no TrainingSession). Use when
# shipping a chat-only deployment.
wasm-pack build crates/rullama --target web --release --out-dir ../../pkg

# Native parity / smoke tests against a local Ollama GGUF blob
cargo run -p rullama --release --example greedy_parity -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" 5

cargo run -p rullama --release --example model_api -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --greedy --max=16

cargo run -p rullama --release --example chained_smoke -- \
    ~/.ollama/models/blobs/sha256-<digest> "Hi" --max=8

# Standard Rust hygiene
cargo build --workspace
cargo test  -p rullama
cargo test  -p rullama --test <name>             # single integration test
cargo test  -p rullama <module::test_name>       # single in-module test
cargo clippy --workspace --all-targets
cargo fmt --all
```

`--features cpu-reference` is a **no-op** kept for back-compat with existing
scripts — the f32 oracle is always built. Don't add new uses.

Examples consume a GGUF blob path like `~/.ollama/models/blobs/sha256-<digest>`;
get one via `ollama pull gemma4:e2b` and read the manifest. The full set lives
under `crates/rullama/examples/` (parity, smoke, inspectors, microbenches) and
`crates/rullama-finetune/examples/` (overfit_one, train_jsonl).

## PWA dev loops

Two harnesses against the same `pkg/` bundle:

| Path              | When to use                                                                 |
|-------------------|------------------------------------------------------------------------------|
| `examples/web/`   | User-facing chat PWA work — React + Vite + Tailwind + Workbox SW. `pnpm dev` auto-runs the wasm build. |
| `examples/pwa/`   | Kernel benchmarks and scripted iPhone runs via `safaridriver`. Build the wasm bundle first, then `./examples/pwa/serve.sh` (HTTPS). |

iPhone runs go through `examples/pwa/run-on-iphone.sh` / `iphone-session-keeper.sh`
/ `clean-iphone.sh`. Logs land at `/tmp/rullama-page.log` (beacons:
`[chat]`, `[pe]`, `[tg]`, `[gen]`, `[wkr]`, `[rs]`). After kernel changes,
remember the harness ships **bit-identical** parity vs Ollama on desktop —
verify locally before touching the iPhone path.

## Docker

Use the cargo aliases, not raw `docker compose` — they go through `xtask`:

```sh
cargo docker:build      # docker compose build
cargo docker:start      # docker compose up -d
cargo docker:stop       # docker compose down
cargo docker:restart    # build --no-cache + up -d --force-recreate
cargo docker:logs       # logs -f --tail=200
cargo docker:ps
```

Add a task by appending a match arm in `xtask/src/main.rs` and the alias line in
`.cargo/config.toml`.

## Architectural rules of the road

- **Worker isolation is load-bearing.** Inference runs in a Dedicated Worker
  that owns both the wasm `Model` handle and a `FileSystemSyncAccessHandle` over
  OPFS. iOS Safari only exposes sync OPFS in Worker contexts, and the Worker
  shields inference from the iOS main-thread page-watchdog reaper. Don't move
  Model state to the main thread.
- **Weights never enter wasm linear memory in bulk.** `TensorFetcher`
  (`gguf/fetcher.rs`) does per-tile range fetches — either OPFS sync reads or
  HTTP `Range:` requests. wasm peak should stay in tens of MiB regardless of
  model size. Don't read whole tensors when designing new ops.
- **Every new WGSL kernel pairs with a CPU oracle + GPU-vs-CPU parity test.**
  CPU oracle lives in `reference/`; the parity test lives in
  `crates/rullama/examples/` (or a unit test next to the kernel). This is how
  numerical regressions get caught.
- **Design for edge GPUs first; optimize after correctness.** The current
  matmul kernels are deliberately naive; tiling / bind-group caching / fusion
  is the next milestone. Don't optimize speculatively.
- **One CommandEncoder per transformer layer.** `forward_chained.rs` submits
  per-layer so tight-RAM phones can drain the GPU smoothly; don't collapse this
  back into one big submit.
- **GPU-resident KV cache.** Multi-turn chat and mid-generation stop rely on
  it; don't reintroduce CPU readbacks per token.
- **Public API surface is small on purpose.** Stable modules: `api`,
  `error`, `sampling`, `lora`. If you're adding something for the
  wasm-bindgen / native consumer of `rullama`, it goes through `api::Model`.
  Adapter parsing lives in `lora`. Touching the doc-hidden modules'
  signatures across patch releases is allowed but should be noted in the
  changelog. For `rullama-finetune`, the JS-facing surface is
  `TrainingSession` in `wasm_bindgen_api.rs`.

## Layout pointers

```
crates/rullama/src/
  api.rs                  # JS-facing Model (load / loadFromUrl / loadFromOpfs[TextOnly] / generate / stop / loadAdapter / clearAdapter)
  lora.rs                 # InferenceAdapter — parses the safetensors blob TrainingSession writes
  backend/                # WgpuCtx, dispatchers, pipeline cache, WeightCache
  gguf/                   # GGUF v3 reader, TensorFetcher impls (InMemory / HttpRange / Opfs), dequant
  kernels/wgsl/           # 70+ hand-written compute shaders (text + vision + audio + backward)
  model/config.rs         # Gemma4Config — parses gemma4.* GGUF metadata keys
  multimodal/vision.rs    # ViT (16 blocks, 768d)
  multimodal/audio.rs     # Conformer (12 blocks, 1024d, block-local attention)
  reference/forward.rs    # CPU f32 oracle
  reference/forward_chained.rs  # Production GPU forward (per-layer encoder submits)
  sampling.rs             # temperature / top-k / top-p / rep penalty
  template/gemma4_small.rs # Chat template — matches Ollama's render
  tokenizer/              # GGUF BPE — bit-exact vs Ollama

crates/rullama-finetune/src/
  session.rs              # forward → loss → backward → Adam
  lora.rs                 # per-LoRA GPU state (A / B), grad buffers
  scratch.rs              # per-step GPU scratch buffers for backward
  dataset_loader.rs       # JSONL + Tokenizer trait
  wasm_bindgen_api.rs     # JS-facing TrainingSession (wasm32 only); save/load adapter as safetensors

examples/web/             # React + Vite production PWA
examples/pwa/             # Vanilla JS bench + safaridriver scripts
tools/ios-bench/          # Excluded from workspace; staticlib for Xcode
```

## Known sharp edges

- Greedy output is **not** bit-identical to Ollama on every prompt; OOD inputs
  diverge (e.g. "Once upon a time, there was a"). CPU↔GPU consistency within
  rullama is clean (≤8e-5 max abs). Don't claim "Ollama parity" without naming
  the prompt set.
- iPhone path skips vision/audio towers (text-only loader, `max_context=512`)
  to fit in shared RAM — mobile multimodal is a follow-up.
- `subgroups` ✗ on iOS Safari WebGPU. Vision attention falls through to the
  no-subgroup HPD-f16 kernel automatically; preserve that fallback.
- Don't skip git hooks (`--no-verify`, etc.) without explicit user request.
