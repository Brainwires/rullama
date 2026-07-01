# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**rullama is the consumer product family** — a monorepo of three front-end apps,
all running Gemma 4 on the local GPU (with optional cloud providers):

- **`apps/web/`** — the PWA (React + Vite + Tailwind + Workbox SW). Runs AI **in
  the browser** via the engine's wasm bundle.
- **`apps/native/`** — the desktop + mobile app (.NET / Avalonia) over a Rust
  `rust-core` C-ABI shim that links the engine directly.
- **`apps/cli/`** — the agentic CLI (its own Cargo workspace; BYOK providers,
  tools, MCP).

…plus `services/dev-server/` (the dev/serve HTTP server that hosts the PWA) and
`services/worker/` (the Cloudflare BYOK cloud proxy).

The **inference engine + agent harness live in a separate repo** —
**rullama-framework** — as `rullama-engine` (+ `rullama-lora` for local LoRA
training) in an isolated `engine/` wasm32 sub-workspace, plus the `rullama-*`
harness crates. The Gemma 4 forward pass, WGSL kernels, GGUF, tokenizer,
vision/audio/diffusion towers, image-gen, TTS, embeddings — all engine concerns —
are documented there, not here.

> The engine handles **tokens**; the harness handles **turns**; these apps sit
> **on top**. One brand — **rullama** — across the stack; **Brainwires** is the
> company / GitHub org, not a project name. See the canonical topology doc
> `rullama-framework/docs/ARCHITECTURE-engine-harness.md`.

### How each app consumes the platform

1. **`apps/web` (in-browser):** imports the engine's **wasm bundle** at
   `/pkg/rullama.js` (classes `Model`, `TrainingSession`, `EmbeddingModel`,
   `DiffusionGemma`) in a Dedicated Worker; built from the sibling engine checkout
   (see "Engine bundle" below) — no engine Rust source lives here. Cloud via the
   `services/dev-server` `/api/cloud/*` proxy + `apps/web/src/lib/cloud/*` (BYOK).
2. **`apps/native` (C-ABI):** `rust-core` links `rullama-engine` + `rullama-lora`
   directly via P/Invoke (no HTTP, no wasm).
3. **`apps/cli` (path-deps):** depends on the `rullama-framework` harness crates;
   BYOK providers via `rullama-provider`.
4. **Native serve (optional):** any OpenAI-compatible client can point at the
   engine's `rullama-serve` bin (`POST /v1/chat/completions`).

## Workspace layout

The root Cargo workspace is just `xtask`; everything that pulls the framework
crates / wgpu is **excluded** and built via `--manifest-path` (keeps the root
build native-only + fast).

| Path | Target | Notes |
|------|--------|-------|
| `apps/web/` | PWA (TS) | React + Vite + Tailwind + Workbox SW. Imports the engine wasm bundle from `/pkg/rullama.js`. |
| `apps/native/` | .NET + Rust | `app/` (Avalonia) + `rust-core/` (C-ABI cdylib over the engine). Excluded from the root workspace. |
| `apps/cli/` | native | The agentic CLI — its **own** Cargo workspace (own `Cargo.lock`); path-deps `../../../rullama-framework`. Excluded. |
| `services/dev-server/` | native | The dev/serve HTTP server: Vite proxy, `/api/blob`, `/api/models`, `/api/cloud/*`, `/pkg/*`, cross-repo wasm-bundle watcher. Excluded (axum/tokio/notify); run via `--manifest-path`. |
| `services/worker/` | TS | Cloudflare Worker — production BYOK cloud proxy (deployed via `wrangler`). |
| `xtask` | native | Tiny std-only dispatcher for `cargo dev` + `cargo docker:*`. The root workspace. |
| `pkg/` | wasm bundle | Built artifact from the engine (`--out-name rullama`). Gitignored, at the repo root; sourced at dev/build time. |

Rust toolchain pinned to **1.91** via `rust-toolchain.toml`.

## Engine bundle (the one cross-repo coupling)

The PWA needs the engine's wasm bundle in `pkg/`. It is built from a **sibling
rullama-framework engine checkout**:

```sh
# Resolved automatically; override with RULLAMA_ENGINE_DIR.
# Default: ../rullama-framework/engine  (sibling of this repo)
wasm-pack build rullama-lora --target web --release \
    --out-dir <this-repo>/pkg --out-name rullama   # run inside the engine dir
```

- `cargo dev` (local): the devserver watcher rebuilds the bundle from the engine
  checkout when its source changes (`services/dev-server/src/watcher.rs` +
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
pnpm -C apps/web build
pnpm -C apps/web dev          # (cargo dev wraps this via the Vite proxy)

# devserver standalone (it's excluded from the workspace)
cargo build --manifest-path services/dev-server/Cargo.toml --release

cargo build              # workspace = just xtask now
cargo clippy --workspace --all-targets
cargo fmt --all
```

## PWA dev loops

The user-facing PWA lives in `apps/web/` (React + Vite + Tailwind + Workbox SW),
built against the shared `pkg/` wasm bundle. `cargo dev` runs the devserver and
(when an engine checkout is present) keeps the bundle fresh.

iPhone / safaridriver runs go through `apps/web/serve-iphone.sh` / `apps/web/serve-tunnel.sh`
and `apps/web/test/iphone-test.sh`. Logs land at `/tmp/rullama-page.log` (beacons:
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

`cargo dev` runs the native Rust devserver at `services/dev-server/`. Two modes:

| Mode | Command | Vite proxy? | `/api/log` writeable? | `/api/models` listed? | Use when |
|------|---------|-------------|-----------------------|-----------------------|----------|
| Local dev (default) | `cargo dev` | yes (HMR works through :25321) | yes | yes | working locally, **tunnel is OFF** |
| Public / tunnel-safe | `cargo dev -- --public` | no (serves `apps/web/dist/`) | no | no | tunnel is up, public origin is reachable |

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
  happen in the rullama-framework repo.
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
