# Hand-off note — dev server is live, here's how

You are an agent picking up work on **rullama**. Before you touch anything that runs in the browser, read this — there's a native Rust dev server running on this machine that replaces the Python `serve-*.sh` scripts the older docs mention. Don't restart the Python server. Don't bind `:25321`. Don't touch the Cloudflare tunnel.

## What's running

A native dev server (`crates/rullama-devserver/`) is supervised by **PM2** under the name `rullama-devserver`:

```
pm2 status                    # see it
pm2 logs rullama-devserver    # tail logs
pm2 restart rullama-devserver # restart (after rebuilding the binary)
```

Process model:

```
internet → Cloudflare tunnel → http://127.0.0.1:25321
                                 │
                                 └── PM2 (rullama-devserver, --public mode)
                                       │
                                       ├── serves examples/web/dist/  (the React shell)
                                       ├── serves /pkg/*              (wasm-pack output)
                                       └── /api/blob/*                (range-aware GGUF streaming)
```

Public hostname: `https://rullama.brainwires.net`.

## "--public" mode — what it does and doesn't do

The PM2 entry runs `rullama-devserver --public ...`. The `--public` flag composes safe defaults for the tunnel:

- **Serves `examples/web/dist/`** (prebuilt React app) instead of reverse-proxying Vite. **Vite isn't running on this PM2 instance.** Don't expect HMR to work over the public URL — there's no source code on the public side, only the compiled bundle.
- `/api/log` writes — **disabled** (would be an unauthenticated disk-write primitive).
- `/api/models` listing — **disabled** (would leak the user's local Ollama install).
- `/__rullama-dev-ws` (wasm-rebuild broadcast) — **disabled**.
- `/api/blob/{family}:{tag}` — **enabled** (the PWA needs to download GGUFs; Range + 1 MiB chunked streaming).
- CORS — allow-list only, no wildcard. Currently set to `https://rullama.brainwires.net`. **If you add a new origin** (different domain, a staging tunnel, etc.) edit the `--cors-origins` arg in `ops/pm2/ecosystem.config.cjs` and `pm2 restart`.

Security headers (every response): `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, `Cross-Origin-Resource-Policy: cross-origin` on `/api/blob`, `/api/models`, `/pkg/*` (rest get `same-origin`). The PWA loads in a cross-origin-isolated context (WebGPU + SharedArrayBuffer paths assume it), so don't touch these without understanding why.

## The OTHER mode — `cargo dev` (local-only)

If you need to edit React code with hot-reload, run **locally** (not via PM2):

```sh
pm2 stop rullama-devserver    # free :25321
cargo dev                     # spawns Vite + WASM watcher + axum on :25321
```

This is the "local dev" mode — Vite reverse-proxy is on, /api/log accepts writes, /api/models lists, wasm watcher pings the page on rebuilds. **Do NOT run this with the Cloudflare tunnel pointing at it** — Vite's `fs.allow=[repoRoot]` exposes the whole repo over the wire. When done, `pm2 start rullama-devserver` (or run `ops/pm2/setup.sh`) to put `--public` mode back.

Vite HMR works through `:25321` (WebSocket upgrade proxy was just added) — meaning you can load the page from either `http://localhost:25321` or `http://localhost:5173` and saving a `.tsx` pushes the change live in either case.

WASM auto-rebuild: edit anything in `crates/rullama/src/**` or `crates/rullama-finetune/src/**` → `wasm-pack build` fires in the background → page reloads when it lands.

## Adding a new feature (TTS, for example)

1. **Don't change the public surface without considering security.** Adding a new `/api/foo` route means it's reachable from the internet. Mount it in `crates/rullama-devserver/src/api.rs` (or a sibling module) and gate it on a config knob in `crates/rullama-devserver/src/config.rs` — `--public` mode should disable it unless you explicitly want it public. There's prior art in how `/api/log` and `/api/models` are mounted (see `lib.rs::build_app`).

2. **Rebuild + restart** after backend changes:
   ```sh
   cargo build --manifest-path crates/rullama-devserver/Cargo.toml --release
   pm2 restart rullama-devserver
   ```
   The PM2 ecosystem points at the release binary directly (NOT `cargo run`), so the binary on disk has to exist or PM2 will keep crashing.

3. **Rebuild the PWA** after frontend changes:
   ```sh
   cd examples/web && pnpm exec vite build
   ```
   PM2 doesn't need to restart for dist/ changes — it re-reads files off disk on every request. But the browser's service worker may cache the old bundle; hard-reload after a deploy.

4. **Tests** for the devserver crate:
   ```sh
   cargo test --manifest-path crates/rullama-devserver/Cargo.toml --release
   ```
   Currently 18 integration tests. If you add a new route, add a test (`tests/http_endpoints.rs` is the pattern — tower::ServiceExt::oneshot against the in-process Router, no real TCP bind, fast).

5. **Don't kill cloudflared.** It runs as a launchd-managed daemon, PID changes across reboots, and auto-reconnects to whatever's at `:25321`. Killing it costs the user 5 minutes of manual recovery for no reason.

## Quick smoke-test recipe

```sh
# Is the PM2 instance healthy?
curl -sI http://127.0.0.1:25321/api/blob/gemma4:e2b

# Is the tunnel forwarding correctly?
curl -sI https://rullama.brainwires.net/api/blob/gemma4:e2b

# Does CORS echo the right origin?
curl -sI -H 'Origin: https://rullama.brainwires.net' http://127.0.0.1:25321/api/blob/gemma4:e2b \
    | grep -i access-control-allow-origin
```

All three should return 200 with sensible headers (or 206 if you added a `Range:` header).

## Where things live

| Path | What |
|------|------|
| `crates/rullama-devserver/src/lib.rs` | Router builder, layer wiring |
| `crates/rullama-devserver/src/bin/server.rs` | CLI entry, arg parsing, PM2 binary target |
| `crates/rullama-devserver/src/api.rs` | `/api/*` routes |
| `crates/rullama-devserver/src/pkg.rs` | `/pkg/*` static (wasm-pack output) |
| `crates/rullama-devserver/src/dist.rs` | `/dist/*` static for `--public` mode |
| `crates/rullama-devserver/src/proxy.rs` | Vite reverse-proxy (HTTP + WS upgrade) — local-dev only |
| `crates/rullama-devserver/src/watcher.rs` | notify → wasm-pack on Rust edits |
| `crates/rullama-devserver/src/ws.rs` | `/__rullama-dev-ws` wasm-rebuild broadcast |
| `crates/rullama-devserver/src/security.rs` | COOP/COEP/CORP + CORS middleware |
| `crates/rullama-devserver/src/config.rs` | `SecurityConfig` knobs (the `--public` defaults are here) |
| `crates/rullama-devserver/tests/http_endpoints.rs` | 18 integration tests |
| `ops/pm2/ecosystem.config.cjs` | PM2 process config |
| `ops/pm2/setup.sh` | One-shot bring-up: builds binary + dist, restarts PM2, saves process list |
| `examples/web/src/lib/dev-hmr.ts` | Browser WS client for the wasm-rebuild broadcast (dev-only, tree-shaken from production) |

## Plan reference

The design notes for this dev server (including a security audit, test plan, and PM2 ops) are in `~/.claude/plans/write-this-up-formally-delegated-sun.md`. Worth a skim if you're touching the dev-server crate itself.
