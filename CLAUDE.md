# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Browser-resident AI runtime in pure Rust → WebAssembly + WebGPU. Loads
Ollama's on-disk GGUF blobs (no server) and runs the forward pass on the
local GPU through hand-written WGSL. The scope expands as Ollama's does.

**Currently in scope:**

- **Text / vision / audio-input chat** — Gemma 4 only (`gemma4:e2b`,
  `gemma4:e4b`). `Q4_K_M` mix only (`Q4_K` / `Q6_K` / `F16` / `F32`).
  This is the most mature surface and the parity oracle for everything
  else.
- **In-browser LoRA fine-tuning** — over the same Gemma 4 forward path.
  Production-shaped (no Python toolchain, no server upload) and so far
  has no peer in any other browser-LLM project (see
  [[project-competitive-landscape]] memory).
- **Speech synthesis (TTS)** — Kokoro-82M (StyleTTS2 + iSTFTNet) port to
  Rust/WGSL, in-flight. See `project_tts_kokoro` memory; reference impl
  is hexgrad/kokoro PyTorch.
- **Text embeddings + RAG** — EmbeddingGemma-300M (architecture `gemma3`,
  encoder-only) → `EmbeddingModel` in `embed.rs` over a bidirectional
  CPU oracle (`reference/embed/`), validated at cosine 0.9997 vs Ollama.
  Its GGUF is SentencePiece-unigram (scores, not BPE merges) so it needs
  the `tokenizer::spm` SPM tokenizer. Powers the PWA's Knowledge tab
  (drop/paste docs → chunk → embed → rsqlite-wasm vector store) and
  per/cross-conversation chat RAG. CPU forward ships first; a GPU
  forward + memory-streaming the 621 MB GGUF are the open perf items.

**Planned for future versions (roughly in order):**

- **Image generation** — Ollama added experimental support for FLUX.2
  [klein] and Z-Image-Turbo on 2026-01-20. Pulling those through the
  same `/api/blob/<key>` plumbing is straightforward; the engine work
  (UNet cross-attention, VAE conv kernels, CLIP text encoder) is a
  second engine living alongside Gemma's. Browser prior art exists
  (MLC web-stable-diffusion, MDST Engine, SDTurbo-WebGPU) so this is
  catch-up parity not a moat — but it's table-stakes for the
  "Ollama in your browser" pitch.
- **Image editing** — Ollama-driven, once they ship it natively.
- **Agents** — local multi-step planning + tool use against the
  in-browser model. Roadmapped after image gen lands.

**Explicitly out of scope:**

- Other transformer architectures and MoE variants — porting a second
  text LLM family for its own sake doesn't pay off; the Gemma 4 focus
  is doing real work for the test surface and parity claims.
- Server-side anything. The whole pitch is "your data never leaves
  the device."
- Python in the runtime loop. Native Rust + browser only.

When adding a new model family, follow Kokoro's pattern: a sibling
module under `crates/rullama/src/` (or its own crate if substantial),
sharing the `wgpu`/`bytemuck`/`half` foundation and the
`backend::WgpuCtx` + bind-cache infra, but with its own forward path
and its own WGSL kernels. The Gemma 4 path stays untouched.

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
# Native dev server — one command brings up the full stack with React HMR
# AND Rust → WASM auto-rebuild + browser reload. Replaces the legacy
# Python serve-tunnel.sh / serve-iphone.sh. See "Dev server modes" below.
cargo dev                # local dev (Vite proxy, /api/log open, /api/models open)
cargo dev -- --public    # tunnel-safe (dist/ static serve, hardened defaults)
cargo dev -- --help      # all flags

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

The user-facing PWA lives in `web/` (React + Vite + Tailwind + Workbox SW),
built against the shared `pkg/` wasm bundle. `pnpm dev` (or `cargo dev`)
auto-runs the wasm build.

iPhone / safaridriver runs go through `web/serve-iphone.sh` / `web/serve-tunnel.sh`
and `web/test/iphone-test.sh`. Logs land at `/tmp/rullama-page.log` (beacons:
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

## Dev server modes (replaces `serve-tunnel.sh` / `serve-iphone.sh`)

`cargo dev` runs the native Rust devserver at `crates/rullama-devserver/`. Two modes:

| Mode | Command | Vite proxy? | `/api/log` writeable? | `/api/models` listed? | Use when |
|------|---------|-------------|-----------------------|-----------------------|----------|
| Local dev (default) | `cargo dev` | yes (HMR works through :25321) | yes | yes | working locally, **tunnel is OFF** |
| Public / tunnel-safe | `cargo dev -- --public` | no (serves `web/dist/`) | no | no | tunnel is up, public origin is reachable |

**Important security boundary**: `cargo dev` (no flags) reverse-proxies `*` to Vite. Vite's `fs.allow=[repoRoot]` exposes every file under the repo to whatever can reach :25321 — including, transitively, anyone on the internet via `https://rullama.brainwires.net`. **Run `cargo dev --public` whenever the Cloudflare tunnel is up.**

Headers honored on every response: `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, `Cross-Origin-Resource-Policy: cross-origin` on `/api/blob`/`/api/models`/`/pkg/*` (so the cross-origin-isolated page on the tunnel hostname can fetch them from a localhost origin via `?localBlob=`), `same-origin` elsewhere. CORS is allow-list only (`--cors-origins https://rullama.brainwires.net,…`) — no wildcard.

Tunnel-side hardening recommended (Cloudflare dashboard, not in repo):
- WAF: block POST to anything except `/api/log` (and even that, optionally).
- Rate limit: 100 req/min/IP per route.
- Cloudflare Access: optional zero-trust gating (Google / GitHub auth) in front of the public hostname.

Persistent background hosting via PM2: see `ops/pm2/setup.sh`. One-time bring-up:

```sh
./ops/pm2/setup.sh
sudo pm2 startup launchd -u $USER --hp $HOME    # boot survival, once
pm2 save
# day-to-day:
pm2 logs rullama-devserver
pm2 restart rullama-devserver
pm2 status
```

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

web/                      # React + Vite production PWA (+ safaridriver scripts)
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
