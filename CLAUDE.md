# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**rullama is the app** — the consumer-facing PWA (React + Vite + Tailwind +
Workbox SW in `web/`) plus the native dev/serve server that hosts it
(`crates/rullama-devserver`). It runs AI **in the browser** on the local GPU and
also talks to optional cloud providers.

The **inference engine moved out of this repo.** It now lives in the
**brainwires** platform repo as `brainwires-engine` (+ `brainwires-lora` for
local LoRA training), in an isolated `engine/` wasm32 sub-workspace. The Gemma 4
forward pass, WGSL kernels, GGUF loading, tokenizer, vision/audio/diffusion
towers, image-gen, TTS, embeddings — all engine concerns — are documented in the
engine's own CLAUDE.md, not here.

> The engine handles **tokens**; the harness handles **turns**; this app sits
> **on top**. Two brands: **rullama** = the consumer product family (this app +
> `rullama-cli` + the paid `rullama-native`), **brainwires** = the OSS platform
> (engine + harness). See the canonical topology doc
> `brainwires-framework/docs/ARCHITECTURE-engine-harness.md`.

### How the app consumes the engine

1. **In-browser (primary):** the app imports the engine's **wasm bundle** at
   `/pkg/rullama.js` (classes `Model`, `TrainingSession`, `EmbeddingModel`,
   `DiffusionGemma`), driven inside a Dedicated Worker. The bundle is **built
   from the sibling engine checkout** (see "Engine bundle" below), not from this
   repo — there is no engine Rust source here anymore.
2. **Cloud passthrough:** the devserver `/api/cloud/*` proxy + `web/src/lib/cloud/*`
   relay to OpenAI / Ollama Cloud (BYOK).
3. **Native (optional):** any OpenAI-compatible client can point at the engine's
   `brainwires-serve` bin (`POST /v1/chat/completions`).

## Workspace layout

This repo is now small — the engine left. Two-crate-ish layout:

| Path | Target | Notes |
|------|--------|-------|
| `web/` | PWA (TS) | React + Vite + Tailwind + Workbox SW. Imports the engine wasm bundle from `/pkg/rullama.js` over HTTP at runtime. |
| `crates/rullama-devserver` | native | The dev/serve HTTP server: Vite proxy, `/api/blob`, `/api/models`, `/api/cloud/*`, `/pkg/*`, and the cross-repo wasm-bundle watcher. **Excluded** from the workspace (axum/tokio/notify); run via `--manifest-path`. |
| `xtask` | native | Tiny std-only dispatcher for `cargo dev` + `cargo docker:*`. Keep it dependency-free. |
| `pkg/` | wasm bundle | Built artifact from the engine (`--out-name rullama`). Gitignored; sourced at dev/build time. |

Rust toolchain pinned to **1.91** via `rust-toolchain.toml`.

## Engine bundle (the one cross-repo coupling)

The PWA needs the engine's wasm bundle in `pkg/`. It is built from a **sibling
brainwires engine checkout**:

```sh
# Resolved automatically; override with BRAINWIRES_ENGINE_DIR.
# Default: ../brainwires-framework/engine  (sibling of this repo)
wasm-pack build brainwires-lora --target web --release \
    --out-dir <this-repo>/pkg --out-name rullama   # run inside the engine dir
```

- `cargo dev` (local): the devserver watcher rebuilds the bundle from the engine
  checkout when its source changes (`crates/rullama-devserver/src/watcher.rs` +
  `state.rs::engine_dir`). No engine checkout → it serves a prebuilt `pkg/` as-is.
- `ops/pm2/start.sh` (prod): stale-checks the engine source and rebuilds `pkg/`
  on `pm2 restart`, else serves the prebuilt bundle.
- CI / no-engine deploys: ship a prebuilt `pkg/` (npm/CDN) — see Phase 8 of the
  split plan.

`--out-name rullama` is deliberate: it keeps the app's `/pkg/rullama.js` import
path stable across the engine rename.

## Build / run

```sh
# Native dev server — full stack with React HMR + (cross-repo) WASM rebuild.
cargo dev                # local dev (Vite proxy, /api/log open, /api/models open)
cargo dev -- --public    # tunnel-safe (dist/ static serve, hardened defaults)
cargo dev -- --help      # all flags

# PWA build / typecheck
pnpm -C web build
pnpm -C web dev          # (cargo dev wraps this via the Vite proxy)

# devserver standalone (it's excluded from the workspace)
cargo build --manifest-path crates/rullama-devserver/Cargo.toml --release

cargo build              # workspace = just xtask now
cargo clippy --workspace --all-targets
cargo fmt --all
```

## PWA dev loops

The user-facing PWA lives in `web/` (React + Vite + Tailwind + Workbox SW),
built against the shared `pkg/` wasm bundle. `cargo dev` runs the devserver and
(when an engine checkout is present) keeps the bundle fresh.

iPhone / safaridriver runs go through `web/serve-iphone.sh` / `web/serve-tunnel.sh`
and `web/test/iphone-test.sh`. Logs land at `/tmp/rullama-page.log` (beacons:
`[chat]`, `[pe]`, `[tg]`, `[gen]`, `[wkr]`, `[rs]`). Engine/kernel parity is the
engine repo's concern — verify there before touching the iPhone path.

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

## Dev server modes

`cargo dev` runs the native Rust devserver at `crates/rullama-devserver/`. Two modes:

| Mode | Command | Vite proxy? | `/api/log` writeable? | `/api/models` listed? | Use when |
|------|---------|-------------|-----------------------|-----------------------|----------|
| Local dev (default) | `cargo dev` | yes (HMR works through :25321) | yes | yes | working locally, **tunnel is OFF** |
| Public / tunnel-safe | `cargo dev -- --public` | no (serves `web/dist/`) | no | no | tunnel is up, public origin is reachable |

**Important security boundary**: `cargo dev` (no flags) reverse-proxies `*` to Vite. Vite's `fs.allow=[repoRoot]` exposes every file under the repo to whatever can reach :25321 — including, transitively, anyone on the internet via `https://rullama.brainwires.net`. **Run `cargo dev --public` whenever the Cloudflare tunnel is up.**

Headers honored on every response: `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, `Cross-Origin-Resource-Policy: cross-origin` on `/api/blob`/`/api/models`/`/pkg/*` (so the cross-origin-isolated page on the tunnel hostname can fetch them from a localhost origin via `?localBlob=`), `same-origin` elsewhere. CORS is allow-list only (`--cors-origins https://rullama.brainwires.net,…`) — no wildcard.

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

## Architectural rules of the road (app)

- **Worker isolation is load-bearing.** Inference runs in a Dedicated Worker
  that owns the wasm `Model` handle and a `FileSystemSyncAccessHandle` over OPFS.
  iOS Safari only exposes sync OPFS in Worker contexts. Don't move engine state
  to the main thread.
- **The app never bundles engine Rust source.** It consumes the engine only via
  the wasm bundle (`/pkg/rullama.js`), the cloud proxy, or `/v1`. Engine changes
  happen in the brainwires repo.
- **Keep the public surface small.** The worker ↔ main-thread RPC and the
  `inference.ts` client are the seam; engine API changes flow through the bundle.
- **`xtask` stays std-only.** No deps, so `cargo dev` is fast on a cold tree.

## Known sharp edges

- The wasm bundle is an **external artifact**. If chat/training behaves oddly
  after an engine change, suspect a **stale `pkg/`** before app code — rebuild
  from the engine checkout (`cargo dev` does this on source change; otherwise run
  the `wasm-pack` line above).
- iPhone path skips vision/audio towers (text-only loader, `max_context=512`).
  Never auto-trigger a multi-GB fresh download on a real device.
- Don't skip git hooks (`--no-verify`, etc.) without explicit user request.
